defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph.GPUNode do
  alias Cuda.Template
  alias Cuda.Graph.{Node, NodeProto}

  import Cuda.Compiler.Utils

  defmodule Helpers do
    alias Cuda.Compiler.Context
    alias Cuda.Memory
    use Bitwise
    require Logger

    def offset(ctx, memory, var) do
      shape = Context.find_assign(ctx, [:memory, memory], ctx.path, &has_var?(&1, var))
      shape = with nil <- shape do
        get_in(ctx.assigns, [:memory, memory])
      end
      Memory.offset(shape, var)
    end

    defp has_var?(%Memory{vars: vars}, var) do
      Keyword.has_key?(vars, var)
    end
    defp has_var?(map, var) do
      Map.has_key?(map, var)
    end

    defp current_node_id(ctx) do
      Node.string_id(Map.get(Context.node(ctx) || %{}, :id))
    end

    def kernel(ctx, name, body, opts \\ []) do
      params = [{:pins, :u64, [ptr: true]}] ++ Keyword.get(opts, :args, [])
      params = params |> Enum.map(&param/1) |> Enum.join(", ")
      ".visible .entry #{current_node_id(ctx)}__#{name} (#{params}) {\n" <>
      body <>
      "\n}"
    end

    def include(ctx, module, part \\ :body, opts \\ []) do
      {part, opts} = case part do
        opts when is_list(opts) -> {:body, opts}
        part                    -> {part, opts}
      end
      with {:module, _} <- Code.ensure_loaded(module) do
        case function_exported?(module, :__ptx__, 2) do
          true -> module.__ptx__(part, Keyword.put(opts, :ctx, ctx))
          _    -> ""
        end
      else
        _ -> raise CompileError, description: "Couldn't compile include module #{module}"
      end
    end

    def param({name, type, opts}) do
      space = opts
              |> Keyword.take(~w(const global local shared)a)
              |> Enum.reduce([], fn
                {name, true}, [] -> [".#{name}"]
                _, acc           -> acc
              end)
      align = opts
              |> Keyword.take(~w(align)a)
              |> Enum.reduce([], fn
                {:align, x}, _ when band(x, x - 1) == 0 -> [".align #{x}"]
                _, acc                                  -> acc
              end)
      param = [".param", ".#{type}"] ++
              (if Keyword.get(opts, :ptr) == true, do: [".ptr"], else: []) ++
              space ++
              align ++
              ["#{name}"]
      param |> Enum.join(" ")
    end

    defmacro defkernel(ctx, name, args, opts) do
      body = Keyword.get(opts, :do)
      args = args
             |> Enum.map(&parse_arg/1)
             |> Enum.filter(&is_tuple/1)
             |> Macro.escape
      quote do
        kernel(unquote(ctx), unquote(name), unquote(body), args: unquote(args))
      end
    end
    defmacro defkernel(ctx, name, opts) do
      body = Keyword.get(opts, :do)
      quote do
        kernel(unquote(ctx), unquote(name), unquote(body))
      end
    end

    defp parse_arg(arg, opts \\ [])
    defp parse_arg({name, type}, opts) when is_atom(type) do
      {name, type, opts}
    end
    defp parse_arg({name, {{:., _, [{type, _, x}, opt]}, _, _}}, opts) when is_atom(x) do
      {name, type, [{opt, true} | opts]}
    end
    defp parse_arg({name, {{:., _, [nested, opt]}, _, _}}, opts) do
      parse_arg({name, nested}, [{opt, true} | opts])
    end
    defp parse_arg({name, {:-, _, [{{:., _, [nested, opt]}, _, _}, v]}}, opts) do
      parse_arg({name, nested}, [{opt, v} | opts])
    end
    defp parse_arg({name, {type, _, _}}, opts) when is_atom(type) do
      {name, type, opts}
    end
    defp parse_arg(_, _) do
      nil
    end
  end

  def sources(node, ctx) do
    node = put_pins_shapes(node)
    ctx = Cuda.Compiler.Context.replace_current(ctx, node)
    helpers = [Helpers] ++ Map.get(node.assigns, :helpers, [])
    opts = [context: ctx, helpers: helpers]
    ptx = case node.module.__ptx__(node) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    ptx = ptx |> Enum.map(& Template.ptx_preprocess(&1, opts))
    c = case node.module.__c__(node) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    c = c |> Enum.map(& Template.c_preprocess(&1, opts))
    {:ok, NodeProto.assign(node, :sources, ptx ++ c)}
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph.GPUNode do
  alias Cuda.Compiler
  alias Cuda.Compiler.{Context, GPUUnit}
  alias Cuda.Graph.{Node, NodeProto}
  require Logger

  def compile(node, ctx) do
    ctx = Context.for_node(ctx, node)
    Logger.info("CUDA: Compiling GPU code for node #{node.module} (#{node.id})")
    with {:ok, node}  <- node.module.__compile__(node),
         {:ok, node}  <- GPUUnit.sources(node, ctx),
         {:ok, cubin} <- Compiler.compile(node.assigns.sources) do
      batch = node.module.__batch__(node)
              |> Enum.map(fn
                {:run, {name, g, b, args}} ->
                  {:run, {"#{Node.string_id(node.id)}__#{name}", g, b, args}}
                {:run, {name, g, b}} ->
                  {:run, {"#{Node.string_id(node.id)}__#{name}", g, b, []}}
              end)
      node = node
             |> NodeProto.assign(:cubin, cubin)
             |> NodeProto.assign(:batch, batch)
      {:ok, node}
    else
      _ ->
        Logger.warn("CUDA: Error occured while compiling GPU code for node " <>
                    "#{node.module} (#{node.id})")
        {:error, :compile_error}
    end
  end
end

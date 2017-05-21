defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph.GPUNode do
  alias Cuda.Template
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Node

  defmodule Helpers do
    use Bitwise
    require Logger

    def pin_offset(ctx, name) do
      with nil <- get_in(ctx.assigns, [:pin_offsets, name]) do
        Logger.warn("Can't find offset for pin `#{name}` of node `#{ctx.node.id}`")
        nil
      end
    end

    def shared_offset(ctx, name) do
      offsets = ctx.assigns.shared_offsets
      with nil <- find_shared_offset(ctx.node.id, name, offsets),
           nil <- find_shared_offset(ctx.node.assigns[:alias], name, offsets) do
        Logger.warn("Can't find offset for shared `#{name}` of node `#{ctx.node.id}`")
        #Logger.warn("Avaialable shares are: #{inspect offsets}")
        nil
      end
    end

    defp find_shared_offset(nil, _, _), do: nil
    defp find_shared_offset(id, name, offsets) do
      id
      |> Node.string_id()
      |> String.split("__")
      |> Enum.reduce([], fn
        part, []               -> [[part]]
        part, [path | _] = acc -> [(path ++ [part]) | acc]
      end)
      |> Enum.map(& Enum.join(&1, "__"))
      |> Enum.find_value(& offsets[name][&1])
    end

    def kernel(ctx, name, body, opts \\ []) do
      params = [{:pins, :u64, [ptr: true]}] ++ Keyword.get(opts, :args, [])
      params = params |> Enum.map(&param/1) |> Enum.join(", ")
      ".visible .entry #{Node.string_id(ctx.node.id)}__#{name} (#{params}) {\n" <>
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
    vars = Map.get(node.assigns, :vars, %{}) |> Enum.into(%{})
    helpers = [Helpers] ++ Map.get(node.assigns, :helpers, [])
    opts = [context: %{ctx | node: node, vars: vars}, helpers: helpers]
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
  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Pin
  alias Cuda.Graph.Node
  require Logger

  import Cuda.Compiler.Utils

  def compile(node, ctx) do
    Logger.info("CUDA: Compiling GPU code for node #{node.module} (#{node.id})")
    with {:ok, node}  <- node.module.__compile__(node),
         %{} = node   <- assign_offsets(node),
         %{} = node   <- put_pins_shapes(node),
         ctx = %{ctx | assigns: Map.put(ctx.assigns, :pin_offsets, node.assigns.pin_offsets)},
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

  defp assign_offsets(%{pins: pins} = node) do
    {offsets, _} = Enum.reduce(pins, {%{}, 0}, fn pin, {offsets, offset} ->
      size = Pin.data_size(pin)
      {Map.put(offsets, pin.id, offset), offset + size}
    end)
    NodeProto.assign(node, :offsets, offsets)
  end
end

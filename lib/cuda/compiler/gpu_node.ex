defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph.GPUNode do
  alias Cuda.Template

  defmodule Helpers do
    use Bitwise

    def offset(ctx, name) do
      get_in(ctx.assigns, [:offsets, name])
    end

    def kernel(ctx, name, body, opts \\ []) do
      params = [{:pins, :u64, [ptr: true]}] ++ Keyword.get(opts, :args, [])
      params = params |> Enum.map(&param/1) |> Enum.join(", ")
      ".visible .entry #{ctx.node.id}__#{name} (#{params}) {\n" <>
      body <>
      "\n}"
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
                  #{:test, {{:., [], [{:me, [], Elixir}, :ptr]}
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
    vars = node.module.__vars__(node.assigns.options, ctx)
    helpers = node.module.__helpers__(node.assigns.options, ctx)
    opts = [context: %{ctx | node: node},
            helpers: [Helpers] ++ helpers,
            vars: vars |> Enum.into(%{})]
    ptx = case node.module.__ptx__(node.assigns.options, ctx) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    ptx = ptx |> Enum.map(& Template.ptx_preprocess(&1, opts))
    c = case node.module.__c__(node.assigns.options, ctx) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    c = c |> Enum.map(& Template.c_preprocess(&1, opts))
    ptx ++ c
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph.GPUNode do
  alias Cuda.Compiler
  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.NodeProto
  require Logger

  def compile(node, ctx) do
    sources = GPUUnit.sources(node, ctx)
    Logger.info("CUDA: Compiling GPU code for node #{node.module} (#{node.id})")
    with {:ok, cubin} <- Compiler.compile(sources) do
      {:ok, NodeProto.assign(node, :cubin, cubin)}
    else
      _ ->
        Logger.warn("CUDA: Error occured while compiling GPU code for node " <>
                    "#{node.module} (#{node.id})")
        {:error, :compile_error}
    end
  end
end

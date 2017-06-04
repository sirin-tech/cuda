defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph.GPUNode do
  alias Cuda.Compiler.Context
  alias Cuda.{Template, Template.PtxHelpers}
  alias Cuda.Graph.{Node, NodeProto}

  import Cuda.Compiler.Utils

  def sources(node, ctx) do
    node = put_pins_shapes(node)
    ctx = Context.replace_current(ctx, node)
    helpers = [PtxHelpers] ++ Map.get(node.assigns, :helpers, [])
    opts = [context: ctx, helpers: helpers]
    ptx = case node.module.__ptx__(node) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    ptx = ptx
          |> Enum.map(& Template.ptx_preprocess(&1, opts))
          |> Enum.map(&include_header(ctx, &1))
    c = case node.module.__c__(node) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    c = c |> Enum.map(& Template.c_preprocess(&1, opts))
    {:ok, NodeProto.assign(node, :sources, ptx ++ c)}
  end

  @line_re ["\n", "\r\n", "\n\r"]
  @space_re ~r/\s+/
  @header_directives ~w(.version .target .address_size)
  defp include_header(ctx, {:ptx, src}) do
    directives = src
                 |> String.split(@line_re)
                 |> Enum.map(&String.trim/1)
                 |> Enum.map(&String.split(&1, @space_re))
                 |> Enum.map(&List.first/1)
                 |> Enum.filter(& &1 in @header_directives)
    src = if ".address_size" in directives do
      src
    else
      PtxHelpers.address_size(ctx) <> src
    end
    src = if ".target" in directives do
      src
    else
      PtxHelpers.target(ctx) <> src
    end
    src = if ".version" in directives do
      src
    else
      PtxHelpers.version() <> src
    end
    {:ptx, src}
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph.GPUNode do
  alias Cuda.Compiler
  alias Cuda.Compiler.{Context, GPUUnit}
  alias Cuda.Graph.{Node, NodeProto}
  require Logger

  def compile(node, ctx) do
    Logger.info("CUDA: Compiling GPU code for node #{node.module} (#{node.id})")
    with {:ok, node}  <- node.module.__compile__(node),
         ctx = Context.for_node(ctx, node),
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

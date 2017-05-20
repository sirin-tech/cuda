defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph do
  alias Cuda.Graph.Node
  alias Cuda.Graph.Pin
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.GraphProto
  alias Cuda.Graph.Processing
  alias Cuda.Compiler.GPUUnit

  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  def sources(%{type: :computation_graph, id: gid} = graph, ctx) do
    # temporary guard to deny graph with multiple inputs and outputs
    # remove it when multiple io will be supported by runner
    if length(NodeProto.pins(graph, input_pin_types())) > 1 do
      raise CompileError, description: "Multiple inputs are not supported"
    end
    if length(NodeProto.pins(graph, output_pin_types())) > 1 do
      raise CompileError, description: "Multiple outputs are not supported"
    end

    with {:ok, nodes} <- Processing.topology_sort(graph) do
      nodes = nodes
              |> Enum.map(fn {node, _pin} -> node end)
              |> Enum.reject(& &1 == gid)
      {_, size1, size2} = nodes
                          |> Enum.with_index
                          |> Enum.reduce({graph, 0, 0}, &collect_sizes/2)
      graph = NodeProto.assign(graph, :pin_size, size1 + size2)

      offset = if rem(length(nodes), 2) == 0, do: 0, else: size1
      offsets = graph.pins |> Enum.reduce(%{}, fn
        %{id: id, type: type}, acc when type in input_pin_types() -> Map.put(acc, id, 0)
        %{id: id, type: type}, acc when type in output_pin_types() -> Map.put(acc, id, offset)
      end)
      graph = NodeProto.assign(graph, :pin_offsets, offsets)

      if Map.get(ctx.assigns, :compile_sources) != false do
        state = {:ok, {graph, 0, size1, ctx, [], []}}
        with {:ok, {_, _, _, _, sources, batch}} <- Enum.reduce(nodes, state, &collect_sources/2) do
          graph = graph
                  |> NodeProto.assign(:sources, sources)
                  |> NodeProto.assign(:batch, batch)
          {:ok, graph}
        else
          {:error, _} = error -> error
          error               -> {:error, error}
        end
      else
        {:ok, graph}
      end
    else
      error -> {:error, error}
    end
  end
  def sources(graph, _), do: {:ok, graph}

  defp collect_sizes({node, idx}, {graph, s1, s2}) do
    node = GraphProto.node(graph, node)
    {pins1, pins2} = case div(idx, 2) do
      0 -> {input_pin_types(), output_pin_types()}
      _ -> {output_pin_types(), input_pin_types()}
    end
    size1 = node
            |> NodeProto.pins(pins1)
            |> Enum.map(&Pin.data_size/1)
            |> Enum.max()
    size2 = node
            |> NodeProto.pins(pins2)
            |> Enum.map(&Pin.data_size/1)
            |> Enum.max()
    {graph, Enum.max([s1, size1]), Enum.max([s2, size2])}
  end

  defp collect_sources(node, {:ok, {graph, offset1, offset2, ctx, sources, batch}}) do
    node = GraphProto.node(graph, node)
    {offsets, _} = node
                   |> NodeProto.pins(input_pin_types())
                   |> Enum.reduce({%{}, offset1}, &collect_offsets/2)
    {offsets, _} = node
                   |> NodeProto.pins(output_pin_types())
                   |> Enum.reduce({offsets, offset2}, &collect_offsets/2)
    assigns = Map.put(ctx.assigns, :offsets, offsets)
    with {:ok, node} <- node.module.__compile__(node),
         {:ok, node} <- GPUUnit.sources(node, %{ctx | assigns: assigns}) do
      node_batch = node.module.__batch__(node) |> Enum.map(fn
                     {name, g, b, args} -> {"#{Node.string_id(node.id)}__#{name}", g, b, args}
                     {name, g, b}       -> {"#{Node.string_id(node.id)}__#{name}", g, b, []}
                   end)
      batch = batch ++ node_batch
      sources = sources ++ node.assigns.sources
      {:ok, {graph, offset2, offset1, ctx, sources, batch}}
    end
  end
  defp collect_sources(_, error), do: error

  defp collect_offsets(pin, {offsets, offset}) do
    {Map.put(offsets, pin.id, offset), offset + Pin.data_size(pin)}
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph do
  alias Cuda.Compiler
  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.GraphProto
  alias Cuda.Graph.Processing
  require Logger

  def compile(%{type: :computation_graph} = graph, ctx) do
    Logger.info("CUDA: Compiling GPU code for graph #{graph.module} (#{graph.id})")
    with {:ok, graph} <- graph.module.__compile__(graph),
         {:ok, graph} <- GPUUnit.sources(graph, ctx) do
      if Map.get(ctx.assigns, :compile_sources) != false do
        with {:ok, cubin} <- Compiler.compile(graph.assigns.sources) do
          {:ok, NodeProto.assign(graph, :cubin, cubin)}
        end
      else
        {:ok, graph}
      end
    else
      _ ->
        Logger.warn("CUDA: Error while compiling GPU code for graph " <>
                    "#{graph.module} (#{graph.id})")
        {:error, :compile_error}
    end
  end
  def compile(graph, ctx) do
    Logger.info("CUDA: Compiling graph #{graph.module} (#{graph.id})")
    with {:ok, graph}    <- graph.module.__compile__(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/source.svg"),
         %{} = graph     <- Processing.expand(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/expanded.svg"),
         %{} = graph     <- Processing.precompile_wrap(graph),
         Cuda.Graph.Visualize.Dot.render(graph, output: "/tmp/wrapped.svg"),
         %{} = graph     <- topology_sort(graph),
         {:ok, _, nodes} <- Enum.reduce(graph.nodes, {:ok, ctx, []}, &compile_reducer/2) do
      assigns = Enum.reduce(nodes, graph.assigns, fn
        %{type: :computation_graph, assigns: %{pin_offsets: offsets}}, acc ->
          offsets = offsets
                    |> Enum.map(fn {pin, value} ->
                      {convert_pin_name(pin, graph.links), value}
                    end)
                    |> Enum.into(%{})
          Map.put(acc, :pin_offsets, offsets)
        _, acc ->
          acc
      end)
      {:ok, %{graph | nodes: nodes, assigns: assigns}}
    else
      _ ->
        Logger.warn("CUDA: Error while compiling graph " <>
                    "#{graph.module} (#{graph.id})")
        {:error, :compile_error}
    end
  end

  defp topology_sort(%{id: gid} = graph) do
    with {:ok, nodes} <- Processing.topology_sort(graph) do
      nodes = nodes
              |> Enum.map(fn {node, _pin} -> node end)
              |> Enum.reject(& &1 == gid)
              |> Enum.map(& GraphProto.node(graph, &1))
      %{graph | nodes: nodes}
    else
      _ -> graph
    end
  end

  defp compile_reducer(node, {:ok, ctx, nodes}) do
    with {:ok, node} <- Cuda.Compiler.Unit.compile(node, ctx) do
      {:ok, ctx, [node | nodes]}
    end
  end
  defp compile_reducer(_, error) do
    error
  end

  defp convert_pin_name(id, links) do
    Enum.find_value(links, fn
      {{:__self__, pin}, {_, ^id}} -> pin
      {{_, ^id}, {:__self__, pin}} -> pin
      _ -> nil
    end) || id
  end
end

defimpl Cuda.Compiler.Unit, for: Cuda.Graph.Node do
  require Logger

  def compile(node, ctx) do
    Logger.info("CUDA: Compiling node #{node.module} (#{node.id})")
    node.module.__compile__(node, ctx)
  end
end

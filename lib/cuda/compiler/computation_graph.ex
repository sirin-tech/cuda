defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph do
  alias Cuda.Graph.Node
  alias Cuda.Graph.Pin
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.GraphProto
  alias Cuda.Graph.Processing

  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  def sources(%{type: :computation_graph, id: gid} = graph, ctx) do
    with {:ok, nodes} <- Processing.topology_sort(graph) do
      nodes = nodes
              |> Enum.map(fn {node, _pin} -> node end)
              |> Enum.reject(& &1 == gid)
      {_, offset, _} = nodes
                       |> Enum.with_index
                       |> Enum.reduce({graph, 0, 0}, &collect_sizes/2)
      state = {graph, 0, offset, ctx, []}
      {_, _, _, _, sources} = nodes |> Enum.reduce(state, &collect_sources/2)
      sources
    else
      _ -> []
    end
  end
  def sources(_, _), do: []

  defp collect_sizes({node, idx}, {graph, s1, s2}) when div(idx, 2) == 0 do
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

  defp collect_sources(node, {graph, offset1, offset2, ctx, sources}) do
    node = GraphProto.node(graph, node)
    {offsets, _} = node
                   |> NodeProto.pins(input_pin_types())
                   |> Enum.reduce({%{}, offset1}, &collect_offsets/2)
    {offsets, _} = node
                   |> NodeProto.pins(output_pin_types())
                   |> Enum.reduce({offsets, offset2}, &collect_offsets/2)
    var = Map.put(ctx.var, :offsets, offsets)
    sources = sources ++ Cuda.Compiler.GPUUnit.sources(node, %{ctx | var: var})
    {graph, offset2, offset1, ctx, sources}
  end

  defp collect_offsets(pin, {offsets, offset}) do
    {Map.put(offsets, pin.id, offset), offset + Pin.data_size(pin)}
  end
end

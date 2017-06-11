defmodule Cuda.Graph.Visualize.Dot do
  alias Cuda.Graph.Node
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.Processing
  import Node, only: [input_pin_types: 0, output_pin_types: 0]

  def render(graph, opts \\ []) do
    gv = render_node(graph)
    n = UUID.uuid1()
    file = Path.join(System.tmp_dir!, "#{n}.gv")
    File.write(file, gv)
    #IO.puts(gv)
    out = Keyword.get(opts, :output, "#{n}.svg")
    System.cmd("dot", ["-Tsvg", file, "-o", out])
    File.rm_rf!(file)
  end

  defp render_pin(pin, node_id) do
    pin_id = node_id(pin.id, node_id)
    label = if is_nil(pin.group), do: "#{pin.id}", else: "#{pin.id} (#{pin.group})"
    ~s(#{pin_id}[label="#{label}"])
  end

  defp render_node(node, parent_id \\ nil) do
    id = node_id(node.id, parent_id)

    color = case node.type do
      :gpu -> "blue"
      :computation_graph -> "green"
      _ -> "lightgrey"
    end

    g = if parent_id == nil, do: "digraph", else: "subgraph"

    i = node
        |> NodeProto.pins(input_pin_types())
        |> Enum.map(& render_pin(&1, id))
        |> Enum.join("; ")
    i = "subgraph #{id}_inputs_cluster {rankdir=TB;#{i}}"

    o = node
        |> NodeProto.pins(output_pin_types())
        |> Enum.map(& render_pin(&1, id))
        |> Enum.join("; ")
    o = case o do
      "" -> ""
      o  -> o <> ";"
    end

    links = if parent_id == nil, do: render_links(node), else: []
    links = links |> Enum.join("; ")

    children = case Map.get(node, :nodes) do
      nil -> ""
      nodes -> nodes |> Enum.map(& render_node(&1, id)) |> Enum.join("\n")
    end

    layout = if parent_id == nil do
      "rankdir=LR;rank=source;"
    else
      "rankdir=LR;rank=source;"
    end

    """
    #{g} "cluster_#{id}" {
      #{layout}
      label="#{Node.string_id(node.id)}";
      color=#{color};
      shape=box;
      #{i};
      #{children}
      #{links}
      #{o}
    }
    """
  end

  defp render_links(%{id: gid} = node, parent_id \\ nil) do
    id = node_id(node.id, parent_id)
    result = Processing.dfs(node, fn
      :enter, {%{nodes: _} = src, _}, st ->
        if node_id(src.id, parent_id) in st.nodes do
          {:ok, st}
        else
          links = render_links(src, id)
          {:ok, %{st | nodes: [src.id | st.nodes], links: st.links ++ links}}
        end
      :enter, {src, _}, st ->
        if node_id(src.id, parent_id) in st.nodes do
          {:ok, st}
        else
          {:ok, %{st | nodes: [src.id | st.nodes]}}
        end
      :move, {{src, src_pin}, {dst, dst_pin}}, st ->
        sp = if src.id == gid, do: parent_id, else: node_id(gid, parent_id)
        dp = if dst.id == gid, do: parent_id, else: node_id(gid, parent_id)
        l = "#{node_id({src.id, src_pin.id}, sp)} -> #{node_id({dst.id, dst_pin.id}, dp)}"
        {:ok, %{st | links: [l | st.links]}}
      _, _, st ->
        {:ok, st}
    end, %{nodes: [id], links: [], path: nil})
    with {:ok, st} <- result, do: st.links |> Enum.uniq
  end

  defp node_id(id, nil), do: Node.string_id(id) |> String.replace("-", "")
  defp node_id(id, parent_id), do: Node.string_id({parent_id, id}) |> String.replace("-", "")
end

defprotocol Cuda.Graph.Processing do
  alias Cuda.Graph
  alias Cuda.Graph.Pin

  @spec dfs(graph :: Graph.t, callback :: Graph.dfs_callback, state :: any) :: Graph.dfs_result
  def dfs(graph, callback, state)

  @spec loop?(graph :: Graph.t) :: boolean
  def loop?(graph)

  @spec expand(graph :: Graph.t) :: Graph.t
  def expand(graph)

  @spec topology_sort(graph :: Graph.t) :: {:ok, [{Graph.id, Graph.id}]}
  def topology_sort(graph)

  @doc """
  Finds longest chains of specific node type all over the graph
  """
  @spec longest_chain(graph :: Graph.t, node_type :: Node.type) :: [[Graph.Node.t]]
  def longest_chain(graph, node_type)

  @doc """
  Move node from source graph into destination graph, when destination graph
  is nested into source graph and moving node belongs to source graph
  """
  @spec move(source_graph :: t, destination_graph :: t, node_id :: term) :: t
  def move(srcg, _dstg, nid)
end

defimpl Cuda.Graph.Processing, for: Cuda.Graph do
  alias Cuda.Graph
  alias Cuda.Graph.Pin
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.GraphProto

  require Cuda
  import Cuda, only: [compile_error: 1]

  @input_pins  ~w(input consumer)a
  @output_pins ~w(output producer)a

  # ----------------------------------------------------------------------------
  # dfs
  # ----------------------------------------------------------------------------
  def dfs(graph, callback, state \\ %{})
  def dfs(%Graph{} = graph, callback, state) do
    st = %{graph: graph, nodes: [], callback: callback, state: state}
    result =
      graph
      |> NodeProto.pins(@input_pins)
      |> Enum.reduce({:ok, st}, fn
        %Pin{id: id}, {:ok, st} -> dfs_search({:__self__, id}, {:ok, st})
        _, result               -> result
      end)
    case result do
      {:error, _} = err -> err
      {action, st} -> {action, st.state}
      result       -> compile_error("Unexpected result `#{inspect result}` returned from `dfs`")
    end
  end

  defp dfs_search(node_spec, {:ok, st}) do
    # get node from graph by node_spec
    node = dfs_node_pin_by_spec(st.graph, node_spec)

    # yield callback if this is a first loopkup of node
    result = if not node_spec in st.nodes do
      with {:ok, st} <- dfs_yield(:enter, node, st) do
        {:ok, %{st | nodes: [node_spec | st.nodes]}}
      end
    else
      {:ok, st}
    end

    # find next available node or leave current if there are no nodes
    with {:ok, st} <- result do
      result = node_spec
               |> dfs_next_spec(st.graph)
               |> Enum.reduce({:ok, st}, & dfs_reducer(node, &1, &2))
      with {:ok, st} <- result, do: dfs_yield(:leave, node, st)
    end
  end
  defp dfs_search(_, result), do: result

  defp dfs_reducer(node, next, {:ok, st}) do
    next = Enum.filter(st.graph.links, fn
      {^next, dst} -> dst
      _            -> nil
    end)
    if next == [], do: compile_error("unconnected pin detected")
    next |> Enum.reduce({:ok, st}, & dfs_next_reducer(node, &1, &2))
  end
  defp dfs_reducer(_node, _next, result) do
    result
  end

  defp dfs_next_reducer(node, {_, dst_spec}, {:ok, st}) do
    if not dst_spec in st.nodes do
      dst = dfs_node_pin_by_spec(st.graph, dst_spec)
      with {:ok, st} <- dfs_yield(:move, {node, dst}, st) do
        # recursion
        dfs_search(dst_spec, {:ok, st})
      end
    else
      {:ok, st}
    end
  end
  defp dfs_next_reducer(_node, _next, result), do: result

  defp dfs_next_spec({:__self__, pin} = node_spec, graph) do
    case NodeProto.pin(graph, pin) do
      %Pin{type: :input} -> [node_spec]
      _                  -> []
    end
  end
  defp dfs_next_spec({node_id, _}, graph) do
    case GraphProto.node(graph, node_id) do
      nil ->
        []
      node ->
        @output_pins
        |> Enum.flat_map(& NodeProto.pins(node, &1))
        |> Enum.map(& {node_id, &1.id})
    end
  end

  defp dfs_yield(action, arg, st) do
    case st.callback.(action, arg, st.state) do
      {action, state} ->
        {action, %{st | state: state}}
      _ ->
        compile_error("Unexpected result returned from `dfs` callback")
    end
  end

  defp dfs_node_pin_by_spec(graph, {:__self__, pin}) do
    {graph, NodeProto.pin(graph, pin)}
  end
  defp dfs_node_pin_by_spec(graph, {node, pin}) do
    node = GraphProto.node(graph, node)
    {node, NodeProto.pin(node, pin)}
  end


  # ----------------------------------------------------------------------------
  # loop?
  # ----------------------------------------------------------------------------
  def loop?(graph) do
    result = dfs(graph, fn
      # graph input visited - reset chain
      :enter, {%Graph{}, %{type: type}}, _ when type in @input_pins ->
        {:ok, MapSet.new()}
      # graph output visited - skip
      :enter, {%Graph{}, _}, chain ->
        {:ok, chain}
      # node visited - add it to chain
      :enter, {%{id: node}, _}, chain ->
        {:ok, MapSet.put(chain, node)}
      # move from graph input - skip
      :move, {{%Graph{}, _}, _}, chain ->
        {:ok, chain}
      # move to graph output - skip
      :move, {_, {%Graph{}, _}}, chain ->
        {:ok, chain}
      # move to node - check if node already in chain
      :move, {_, {%{id: to}, _}}, chain ->
        if MapSet.member?(chain, to) do
          {:stop, true}
        else
          {:ok, chain}
        end
      # leave graph input or output - skip
      :leave, {%Graph{}, _}, chain ->
        {:ok, chain}
      # leave node - pop it from chain
      :leave, {%{id: node}, _}, chain ->
        {:ok, MapSet.delete(chain, node)}
    end, MapSet.new())
    with {:stop, result} <- result do
      result
    else
      _ -> false
    end
  end

  # ----------------------------------------------------------------------------
  # expand
  # ----------------------------------------------------------------------------
  def expand(%{nodes: nodes, links: links} = graph) do
    {nodes, links} = Enum.flat_map_reduce(nodes, links, fn
      %{id: child_id, type: :graph} = child, links ->
        child = child
                |> expand()
        nodes = child.nodes
                |> Enum.map(& expand_id(&1, child_id))
        links = child.links
                |> Enum.map(& expand_link(&1, child_id))
                |> Enum.reduce(links, & expand_link_reducer(&1, &2, child_id))
        {nodes, links}
      node, links ->
        {[node], links}
    end)
    graph = %{graph | nodes: nodes, links: links}
    if loop?(graph), do: compile_error("loop detected in expanded graph")
    graph
  end

  defp expand_link_reducer({{:__self__, spin}, {:__self__, dpin}}, links, gid) do
    {src, dst, links} = Enum.reduce(links, {nil, [], []}, fn
      {{^gid, ^spin} = src, _}, {_, dst, links} -> {src, dst, links}
      {_, {^gid, ^dpin} = d}, {src, dst, links} -> {src, [d | dst], links}
      link, {src, dst, links} -> {src, dst, [link | links]}
    end)
    links ++ Enum.map(dst, & {src, &1})
  end
  defp expand_link_reducer({{:__self__, pin}, dst}, links, gid) do
    links |> replace_dst({gid, pin}, dst)
  end
  defp expand_link_reducer({src, {:__self__, pin}}, links, gid) do
    links |> replace_src({gid, pin}, src)
  end
  defp expand_link_reducer(x, links, _) do
    [x | links]
  end

  defp expand_id(%{id: id} = node, gid) do
    %{node | id: expand_node_id(id, gid)}
  end
  defp expand_id(node, _) do
    node
  end

  defp expand_node_id(t, gid) when is_tuple(t) do
    [gid | Tuple.to_list(t)] |> List.to_tuple
  end
  defp expand_node_id(id, gid) do
    {gid, id}
  end

  defp expand_link({{:__self__, _} = src, {:__self__, _} = dst}, _) do
    {src, dst}
  end
  defp expand_link({{:__self__, _} = src, {dst_node, dst_pin}}, gid) do
    {src, {expand_node_id(dst_node, gid), dst_pin}}
  end
  defp expand_link({{src_node, src_pin}, {:__self__, _} = dst}, gid) do
    {{expand_node_id(src_node, gid), src_pin}, dst}
  end
  defp expand_link({{src_node, src_pin}, {dst_node, dst_pin}}, gid) do
    {{expand_node_id(src_node, gid), src_pin},
     {expand_node_id(dst_node, gid), dst_pin}}
  end

  defp replace_dst(links, from, to) do
    links |> Enum.map(fn
      {src, ^from} -> {src, to}
      x            -> x
    end)
  end

  defp replace_src(links, from, to) do
    links |> Enum.map(fn
      {^from, dst} -> {to, dst}
      x            -> x
    end)
  end

  # ----------------------------------------------------------------------------
  # topology_sort
  # ----------------------------------------------------------------------------
  def topology_sort(graph) do
    with false <- loop?(graph) do
      dfs(graph, fn
        :leave, {%{id: node}, %{id: pin}}, st -> {:ok, [{node, pin} | st]}
        _, _, st                              -> {:ok, st}
      end, [])
    else
      true  -> {:error, :loop}
      error -> error
    end
  end

  # ----------------------------------------------------------------------------
  # longest_chain
  # ----------------------------------------------------------------------------
  def longest_chain(graph, node_type) do
    lchain = graph
    |> expand
    |> NodeProto.pins(@input_pins)
    |> lc_producer_pins(graph)
    |> Enum.reduce([], fn link_part, acc ->
      case lc_link(graph, link_part) do
        [] ->
          acc
        links ->
          chain = links
          |> Enum.reduce([], fn
            ({_, {:__self__, _}}, acc) ->
              acc
            ({_, {node_id, _}}, acc)   ->
              node = GraphProto.node(graph, node_id)
              chain = longest_chain(graph, node_type, node)
              lc_max_list(chain, acc)
          end)
          lc_max_list(chain, acc)
      end
    end)
    if length(lchain) == 0 do
      []
    else
      graph = lc_graph_update(graph, lchain)
      [lchain | longest_chain(graph, node_type)]
    end
  end

  defp longest_chain(graph, type, node, current \\ [], max \\ []) do
    {current, max} = cond do
      !lc_check_inputs?(List.last(current), node, graph) ->
        {[], lc_max_list(current, max)}
      node.type == type ->
        {current ++ [node], max}
      true ->
        {[], lc_max_list(current, max)}
    end
    case lc_out_nodes(graph, node) do
      [] ->
        lc_max_list(current, max)
      outnodes ->
        Enum.reduce(outnodes, [], fn n, acc ->
          chain = longest_chain(graph, type, n, current, max)
          lc_max_list(chain, acc)
        end)
    end
  end

  defp lc_link(%{links: links}, link_part) do
    Enum.filter(links, fn
      {^link_part, _} -> true
      _               -> false
    end)
  end

  defp lc_max_list(list1, list2) do
    length(list1) > length(list2) && list1 || list2
  end

  defp lc_out_nodes(graph, node) do
    node
    |> NodeProto.pins(:output)
    |> Enum.reduce([], fn %Pin{id: id}, acc ->
      case lc_link(graph, {node.id, id}) do
        []    ->
          acc
        links ->
          nodes = links
          |> Enum.reduce([], fn
            {_, {:__self__, _}}, acc -> acc
            {_, {node_id, _}}, acc   -> [GraphProto.node(graph, node_id) | acc]
          end)
          nodes ++ acc
      end
    end)
    |> Enum.uniq()
  end

  defp lc_producer_pins(pin_list, %{nodes: nodes}) do
    pin_list = pin_list != [] && Enum.map(pin_list, &({:__self__, &1.id})) || []
    Enum.reduce(nodes, pin_list, fn node, acc ->
      pins = node
      |> Cuda.Graph.NodeProto.pins(:producer)
      |> Enum.map(&({node.id, &1.id}))
      pins ++ acc
    end)
  end

  defp lc_check_inputs?(nil, _, _), do: true
  defp lc_check_inputs?(%{id: pid}, %{id: cid} = cnode, %{links: links}) do
    inpins = cnode
    |> NodeProto.pins(@input_pins)
    |> length()
    links
    |> Enum.filter(fn
      {{^pid, _}, {^cid, _}} -> true
      _                      -> false
    end)
    |> length() == inpins
  end

  defp lc_graph_update(graph, []), do: graph
  defp lc_graph_update(graph, [node | rest]) do
    index = Enum.find_index(graph.nodes, &(&1.id == node.id))
    nodes = List.update_at(graph.nodes, index, &(%{&1 | type: "longest_chain_stub"}))
    graph = %{graph | nodes: nodes}
    lc_graph_update(graph, rest)
  end

  #-----------------------------------------------------------------------------
  # move
  #-----------------------------------------------------------------------------
  def move(srcg, %{id: dstg_id, nodes: []} = dstg, %{id: nid} = node) do
    {dictpins, dstg} = mv_copy_pins(dstg, node.pins)
    srcg = mv_links_redirect(srcg, dstg_id, nid, dictpins)
    srcg = %{srcg | nodes: List.delete(srcg.nodes, node)}
    dstg = mv_add_node(dstg, node, dictpins)
    nodes = Enum.map(srcg.nodes, fn
      %{id: ^dstg_id} -> dstg
      node            -> node
    end)
    %{srcg | nodes: nodes}
  end
  def move(srcg, %{id: dstg_id} = dstg, %{id: nid} = node) do
    shrdlinks = mv_shared_links(srcg.links, nid, dstg_id)
  end
  def move(srcg, dstg_id, node_id) do
    dstg = mv_get_node(srcg, dstg_id)
    node = mv_get_node(srcg, node_id)
    move(srcg, dstg, node)
  end

  defp mv_get_node(graph, node_id) do
    case GraphProto.node(graph, node_id) do
      nil   -> compile_error("Node #{node_id} not found in graph #{graph.id}")
      node  -> node
    end
  end

  defp mv_add_node(graph, %{id: nid} = node, dict) do
    graph = %{graph | nodes: [node | graph.nodes]}
    links = Map.to_list(dict)
    |> Enum.reduce(graph.links, fn {npid, gpid}, links ->
      %{type: type} = NodeProto.pin(node, npid)
      newlink = cond do
        Enum.member?(@input_pins, type)  -> {{:__self__, gpid}, {nid, npid}}
        Enum.member?(@output_pins, type) -> {{nid, npid}, {:__self__, gpid}}
      end
      [newlink | links]
    end)
    %{graph | links: links}
  end

  defp mv_links_redirect(srcg, dstg_id, node_id, dict) do
    links = Enum.map(srcg.links, fn
      {{^node_id, pid}, other} -> {{dstg_id, dict[pid]}, other}
      {other, {^node_id, pid}} -> {other, {dstg_id, dict[pid]}}
      link                     -> link
    end)
    %{srcg | links: links}
  end

  defp mv_shared_links(links, nid1, nid2) do
    links
    |> Enum.filter(fn
      {{^nid1, _}, {^nid2, _}} -> true
      {{^nid2, _}, {^nid1, _}} -> true
      _                        -> false
    end)
  end

  defp mv_shared_pins(links, trgtid, nghbrid) do
    links
    |> Enum.reduce([], fn
      {{^trgtid, pid}, {^nghbrid, _}}, acc -> [pid | acc]
      {{^nghbrid, _}, {^trgtid, pid}}, acc -> [pid | acc]
      _, acc                               -> acc
    end)
  end

  defp mv_copy_pins(graph, pins) do
    pins
    |> Enum.reduce({%{}, graph}, fn pin, {dict, graph} ->
      id = pin.id
      pin = %{pin | id: UUID.uuid1()}
      graph = %{graph | pins: [pin | graph.pins]}
      dict = Map.put(dict, id, pin.id)
      {dict, graph}
    end)
  end
end

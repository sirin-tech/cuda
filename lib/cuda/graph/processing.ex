defprotocol Cuda.Graph.Processing do
  alias Cuda.Graph
  alias Cuda.Graph.Pin

  @spec dfs(graph :: Graph.t, callback :: Graph.dfs_callback, state :: any, opts :: keyword) :: Graph.dfs_result
  def dfs(graph, callback, state \\ %{}, opts \\ [])

  @spec dfs_reverse(graph :: Graph.t, callback :: Graph.dfs_callback, state :: any) :: Graph.dfs_result
  def dfs_reverse(graph, callback, state)

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
  def longest_chain(graph, node_type \\ :gpu)

  @doc """
  Finds longest path in the graph
  """
  @spec longest_path(graph :: Graph.t) :: [{Graph.id, Graph.id}]
  def longest_path(graph)

  @doc """
  Move node from source graph into destination graph, when destination graph
  is nested into source graph and moving node belongs to source graph
  """
  @spec move(source_graph :: t, destination_graph :: t, node_id :: term) :: t
  def move(srcg, _dstg, nid)

  @doc """
  Finds all longest chains of specific type nodes and wrap each of them into
  computation graphs
  """
  @spec precompile_wrap(graph :: Cuda.Graph.t, node_type :: Cuda.Graph.Node.type) :: Cuda.Graph.t
  def precompile_wrap(graph, node_type \\ :gpu)

  @doc """
  Flattens graph, if it consists of one graph node
  """
  @spec flat(graph :: Cuda.Graph.t) :: Cuda.Graph.t
  def flat(graph)
end

defimpl Cuda.Graph.Processing, for: Cuda.Graph do
  alias Cuda.Graph
  alias Cuda.Graph.Pin
  alias Cuda.Graph.Node
  alias Cuda.Graph.NodeProto
  alias Cuda.Graph.GraphProto

  require Cuda
  import Cuda, only: [compile_error: 1]
  import Node, only: [graph_types: 0, input_pin_types: 0, output_pin_types: 0]

  # ----------------------------------------------------------------------------
  # dfs
  # ----------------------------------------------------------------------------
  def dfs(graph, callback, state \\ %{}, opts \\ [])
  def dfs(%Graph{} = graph, callback, state, opts) do
    st = %{graph: graph,
           nodes: [],
           callback: callback,
           state: state,
           ids: Keyword.get(opts, :ids, false)}
    result =
      graph
      |> NodeProto.pins(input_pin_types())
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
    node = dfs_node_pin_by_spec(st, node_spec)

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
               |> Enum.reduce({:ok, st}, &dfs_reducer/2)
      with {:ok, st} <- result, do: dfs_yield(:leave, node, st)
    end
  end
  defp dfs_search(_, result), do: result

  defp dfs_reducer(next, {:ok, st}) do
    links = Enum.filter(st.graph.links, fn
      {^next, _} -> true
      _          -> nil
    end)
    if links == [], do: compile_error("unconnected pin detected: #{inspect next}")
    links |> Enum.reduce({:ok, st}, &dfs_next_reducer/2)
  end
  defp dfs_reducer(_next, result) do
    result
  end

  defp dfs_next_reducer({src_spec, dst_spec}, {:ok, st}) do
    if not dst_spec in st.nodes do
      src = dfs_node_pin_by_spec(st, src_spec)
      dst = dfs_node_pin_by_spec(st, dst_spec)
      with {:ok, st} <- dfs_yield(:move, {src, dst}, st) do
        # recursion
        dfs_search(dst_spec, {:ok, st})
      end
    else
      {:ok, st}
    end
  end
  defp dfs_next_reducer(_next, result), do: result

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
        output_pin_types()
        |> Enum.flat_map(& NodeProto.pins(node, &1))
        |> Enum.map(& {node_id, &1.id})
    end
  end

  defp dfs_yield(action, arg, st) do
    case st.callback.(action, arg, st.state) do
      {action, state} ->
        {action, %{st | state: state}}
      result ->
        compile_error("Unexpected result `#{inspect result}` returned from `dfs` callback")
    end
  end

  defp dfs_node_pin_by_spec(%{ids: true}, spec), do: spec
  defp dfs_node_pin_by_spec(%{graph: graph}, {:__self__, pin}) do
    {graph, NodeProto.pin(graph, pin)}
  end
  defp dfs_node_pin_by_spec(%{graph: graph}, {node, pin}) do
    node = GraphProto.node(graph, node)
    {node, NodeProto.pin(node, pin)}
  end

  # ----------------------------------------------------------------------------
  # dfs_reverse
  # ----------------------------------------------------------------------------
  def dfs_reverse(graph, callback, state \\ %{})
  def dfs_reverse(%Graph{} = graph, callback, state) do
    st = %{graph: graph, nodes: [], callback: callback, state: state}
    result =
      graph
      |> NodeProto.pins(output_pin_types())
      |> Enum.reduce({:ok, st}, fn
        %Pin{id: id}, {:ok, st} -> dfsr_search({:__self__, id}, {:ok, st})
        _, result               -> result
      end)
    case result do
      {:error, _} = err -> err
      {action, st} -> {action, st.state}
      result       -> compile_error("Unexpected result `#{inspect result}` returned from `dfs`")
    end
  end

  defp dfsr_search(node_spec, {:ok, st}) do
    # get node from graph by node_spec
    node = dfs_node_pin_by_spec(st, node_spec)

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
               |> dfsr_next_spec(st.graph)
               |> Enum.reduce({:ok, st}, &dfsr_reducer/2)
      with {:ok, st} <- result, do: dfs_yield(:leave, node, st)
    end
  end
  defp dfsr_search(_, result), do: result

  defp dfsr_next_spec({:__self__, pin} = node_spec, graph) do
    case NodeProto.pin(graph, pin) do
      %Pin{type: :output} -> [node_spec]
      _                   -> []
    end
  end
  defp dfsr_next_spec({node_id, _}, graph) do
    case GraphProto.node(graph, node_id) do
      nil ->
        []
      node ->
        input_pin_types()
        |> Enum.flat_map(& NodeProto.pins(node, &1))
        |> Enum.map(& {node_id, &1.id})
    end
  end

  defp dfsr_reducer(next, {:ok, st}) do
    next = Enum.filter(st.graph.links, fn
      {_, ^next} -> true
      _          -> nil
    end)
    if next == [], do: compile_error("unconnected pin detected")
    next |> Enum.reduce({:ok, st}, &dfsr_next_reducer/2)
  end
  defp dfsr_reducer(_next, result) do
    result
  end

  defp dfsr_next_reducer({dst_spec, src_spec}, {:ok, st}) do
    if not dst_spec in st.nodes do
      src = dfs_node_pin_by_spec(st, src_spec)
      dst = dfs_node_pin_by_spec(st, dst_spec)
      with {:ok, st} <- dfs_yield(:move, {src, dst}, st) do
        # recursion
        dfsr_search(dst_spec, {:ok, st})
      end
    else
      {:ok, st}
    end
  end
  defp dfsr_next_reducer(_next, result), do: result

  # ----------------------------------------------------------------------------
  # loop?
  # ----------------------------------------------------------------------------
  def loop?(graph) do
    result = dfs(graph, fn
      # graph input visited - reset chain
      :enter, {%Graph{}, %{type: type}}, _ when type in input_pin_types() ->
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
  def expand(%{assigns: %{expanded: x}} = node) when not is_nil(x), do: node
  def expand(%{nodes: nodes, links: links} = graph) do
    {nodes, {links, assigns}} = Enum.flat_map_reduce(nodes, {links, %{}}, fn
      %{id: child_id, type: type} = child, {links, assigns} when type in graph_types() ->
        child = child
                |> expand()
        nodes = child.nodes
                |> Enum.map(& expand_id(&1, child_id))
        links = child.links
                |> Enum.map(& expand_link(&1, child_id))
                |> Enum.reduce(links, & expand_link_reducer(&1, &2, child_id))
        assigns = Map.put(assigns, child_id, child.assigns)
        {nodes, {links, assigns}}
      node, {links, assigns} ->
        assigns = Map.put(assigns, node.id, node.assigns)
        {[node], {links, assigns}}
    end)
    graph = %{graph | nodes: nodes, links: links}
    if loop?(graph), do: compile_error("loop detected in expanded graph")
    graph |> NodeProto.assign(expanded: assigns)
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
  def longest_chain(graph, node_type \\ :gpu) do
    lchain = graph
    |> expand
    |> NodeProto.pins(input_pin_types())
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

  # NOTE: We are temporary disable 2-inputs rule in longest chain
  defp lc_check_inputs?(_, _, _), do: true
  #defp lc_check_inputs?(nil, _, _), do: true
  #defp lc_check_inputs?(%{id: pid}, %{id: cid} = cnode, %{links: links}) do
  #  inpins = cnode
  #  |> NodeProto.pins(input_pin_types())
  #  |> length()
  #  links
  #  |> Enum.filter(fn
  #    {{^pid, _}, {^cid, _}} -> true
  #    _                      -> false
  #  end)
  #  |> length() == inpins
  #end

  defp lc_graph_update(graph, []), do: graph
  defp lc_graph_update(graph, [node | rest]) do
    index = Enum.find_index(graph.nodes, &(&1.id == node.id))
    nodes = List.update_at(graph.nodes, index, &(%{&1 | type: "longest_chain_stub"}))
    graph = %{graph | nodes: nodes}
    lc_graph_update(graph, rest)
  end

  #-----------------------------------------------------------------------------
  # longest_path
  #-----------------------------------------------------------------------------
  def longest_path(%{links: links}) do
    links = links
            |> Enum.filter(&input_link?/1)
            |> Enum.map(& longest_path(links, [], &1))
            |> Enum.filter(&is_list/1)
            |> Enum.sort_by(&length/1)
    case links do
      []    -> []
      links -> List.last(links)
    end
  end
  defp longest_path(_, path, {_, {:__self__, _}} = link) do
    path ++ [link]
  end
  defp longest_path(links, path, {{_src, _}, {dst, _}} = link) do
    output = links
             |> Enum.filter(fn
               {{^dst, _}, _} -> true
               _              -> false
             end)
    if output == [] do
      path ++ [link]
    else
      links = output
              |> Enum.map(& longest_path(links, path ++ [link], &1))
              |> Enum.filter(&is_list/1)
              |> Enum.sort_by(&length/1)
      case links do
        []    -> []
        links -> List.last(links)
      end
    end
  end

  defp input_link?({{:__self__, _}, _}), do: true
  defp input_link?(_), do: false

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
    {srcg, dstg} = mv_common_inputs(srcg, dstg, node)
    {srcg, dstg} = mv_shared_links(srcg, dstg, node)
    {dictpins, dstg} = dstg
    |> mv_copy_pins(mv_pins_to_copy(srcg, dstg, node))
    srcg = mv_links_redirect(srcg, dstg_id, nid, dictpins)
    srcg = %{srcg | nodes: List.delete(srcg.nodes, node)}
    dstg = mv_add_node(dstg, node, dictpins)
    nodes = Enum.map(srcg.nodes, fn
      %{id: ^dstg_id} -> dstg
      node            -> node
    end)
    %{srcg | nodes: nodes}
  end
  def move(srcg, _, []), do: srcg
  def move(srcg, dstg_id, [node_id | rest]) do
    # dstg = mv_get_node(srcg, dstg_id)
    # node = mv_get_node(srcg, node_id)
    # srcg = move(srcg, dstg, node)
    srcg
    |> move(dstg_id, node_id)
    |> move(dstg_id, rest)
  end
  def move(srcg, dstg_id, node_id) when dstg_id != node_id do
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
    links = dict
    |> Map.to_list()
    |> Enum.reduce(graph.links, fn {npid, gpid}, links ->
      %{type: type} = NodeProto.pin(node, npid)
      newlink = cond do
        Enum.member?(input_pin_types(), type)  -> {{:__self__, gpid}, {nid, npid}}
        Enum.member?(output_pin_types(), type) -> {{nid, npid}, {:__self__, gpid}}
      end
      [newlink | links]
    end)
    %{graph | links: links}
  end

  defp mv_links_redirect(srcg, dstg_id, node_id, dict) do
    links = Enum.reduce(srcg.links, [], fn
      {{^node_id, _}, {^dstg_id, _}}, acc -> acc
      {{^dstg_id, _}, {^node_id, _}}, acc -> acc
      {{^node_id, pid}, other}, acc       -> [{{dstg_id, dict[pid]}, other} | acc]
      {other, {^node_id, pid}}, acc       -> [{other, {dstg_id, dict[pid]}} | acc]
      link, acc                           -> [link | acc]
    end)
    %{srcg | links: links}
  end

  defp mv_copy_pins(graph, pins) do
    pins
    |> Enum.reduce({%{}, graph}, fn pin, {dict, graph} ->
      id = pin.id
      pin = %{pin | id: UUID.uuid1()}
      # added by alexiss - when pin type is consumer it should be setted to
      #                    input in graph rather then consumer
      pin = if pin.type == :consumer do
        %{pin | type: :input}
      else
        pin
      end
      graph = %{graph | pins: [pin | graph.pins]}
      dict = Map.put(dict, id, pin.id)
      {dict, graph}
    end)
  end

  defp mv_pins_to_copy(srcg, %{id: dstg_id}, %{id: nid} = node) do
    srcg.links
    |> Enum.reduce([], fn
      {{^nid, _}, {^dstg_id, _}}, acc -> acc
      {{^dstg_id, _}, {^nid, _}}, acc -> acc
      {{^nid, pnid}, _}, acc          -> [NodeProto.pin(node, pnid) | acc]
      {_, {^nid, pnid}}, acc          -> [NodeProto.pin(node, pnid) | acc]
      _, acc                       -> acc
    end)
    # NOTE: added by alexiss. In some cases (training graph) without `Enum.uniq`
    #       produces duplicate pins
    |> Enum.uniq
  end

  defp mv_shared_links(srcg, %{id: dstg_id} = dstg, %{id: nid} = node) do
    shrlinks = srcg.links
    |> Enum.filter(fn
      {{^nid, _}, {^dstg_id, _}} -> true
      {{^dstg_id, _}, {^nid, _}} -> true
      _                          -> false
    end)
    mv_shared_links(srcg, dstg, node, shrlinks)
  end
  defp mv_shared_links(srcg, dstg, _, []), do: {srcg, dstg}
  defp mv_shared_links(srcg, %{id: dstg_id} = dstg, node, [{{nid, npid}, {did, dpid}} = link | rest]) when did == dstg_id do
    pin = NodeProto.pin(dstg, dpid)
    dstg = %{dstg | links: dstg.links
    |> Enum.map(fn
      {{:__self__, ^dpid}, rest} -> {{nid, npid}, rest}
      link                           -> link
    end)}
    dstg = %{dstg | pins: List.delete(dstg.pins, pin)}
    srcg = %{srcg | links: List.delete(srcg.links, link)}
    mv_shared_links(srcg, dstg, node, rest)
  end
  defp mv_shared_links(srcg, %{id: dstg_id} = dstg, node, [{{did, dpid}, {nid, npid}} = link | rest]) when did == dstg_id do
    pin = NodeProto.pin(dstg, dpid)
    count = srcg.links
    |> Enum.filter(fn
      {{^dstg_id, ^dpid}, _} -> true
      _                      -> false
    end)
    |> length()

    dstg = if count == 1 do
      tmp = %{dstg | links: dstg.links
      |> Enum.map(fn
        {other, {:__self__, ^dpid}} -> {other, {nid, npid}}
        link                        -> link
      end)}
      %{tmp | pins: List.delete(tmp.pins, pin)}
    else
      {other, _} = Enum.find(dstg.links, fn
        {_, {:__self__, ^dpid}} -> true
        _                       -> false
      end)
      %{dstg | links: [{other, {nid, npid}} | dstg.links]}
    end

    srcg = %{srcg | links: List.delete(srcg.links, link)}
    mv_shared_links(srcg, dstg, node, rest)
  end

  defp mv_common_inputs(srcg, %{id: did} = dstg, %{id: node_id} = node) do
    # Ищем по исходному графу линки которые идут на вход графа назначения
    dlinks = Enum.reduce(srcg.links, %{}, fn
      {{nid, npid}, {^did, dpid}}, acc when nid != node_id ->
        Map.put(acc, {nid, npid}, {:__self__, dpid})
      _, acc ->
        acc
    end)
    if (dlinks |> Map.keys() |> length()) > 0,
      do: mv_common_inputs(srcg, dstg, node, dlinks),
    else: {srcg, dstg}
  end
  defp mv_common_inputs(srcg, dstg, %{id: node_id} = node, dlinks) do
    # Ищем по исходному графу линки идущие на вход ноды от выходов, которые
    # так же соединены с входами графа назначения
    common = Enum.reduce(srcg.links, [], fn
      {other, {^node_id, _}} = link, acc ->
        if Map.has_key?(dlinks, other)  do
          [link | acc]
        else
          acc
        end
      _, acc ->
        acc
    end)
    mv_common_inputs(srcg, dstg, node, dlinks, common)
  end
  defp mv_common_inputs(srcg, dstg, _, _, []), do: {srcg, dstg}
  defp mv_common_inputs(srcg, dstg, _, dlinks, common) do
    # Удаляем из исходного графа общие линки
    srcg = %{srcg | links: mv_list_remove(srcg.links, common)}
    # Добавляем линки в граф назначения, с перенаправлением на
    # его существующие пины
    dstg = %{dstg | links: Enum.reduce(common, dstg.links, fn {key, other}, acc ->
      [{dlinks[key], other} | acc]
    end)}
    {srcg, dstg}
  end

  defp mv_list_remove(list, []), do: list
  defp mv_list_remove(list, rm) do
    Enum.reduce(rm, list, & List.delete(&2, &1))
  end

  #-----------------------------------------------------------------------------
  # precompile_wrap
  #-----------------------------------------------------------------------------
  def precompile_wrap(graph, node_type \\ :gpu) do
    Code.ensure_loaded(Graph.ComputationGraph)
    chains = graph
    |> longest_chain(node_type)
    |> prc_nodes2ids()
    prc_wrap(graph, chains)
  end

  defp prc_wrap(graph, []), do: graph
  defp prc_wrap(graph, [chain | rest]) do
    nested_id = "comp_" <> UUID.uuid1()
    nested = Graph.Factory.new(%Cuda.Graph{}, nested_id, Graph.ComputationGraph, [], [])

    assigns = Enum.reduce(chain, graph.assigns, fn
      id, assigns when is_tuple(id) ->
        key = id |> Tuple.to_list |> List.first
        case get_in(assigns, [:expanded, key]) do
          nil ->
            assigns
          values ->
            expanded = assigns.expanded
                       |> Map.drop([key])
                       |> Map.put(nested_id, %{expanded: %{key => values}})
            Map.put(assigns, :expanded, expanded)
        end
      _, assigns ->
        assigns
    end)
    graph = %{graph | assigns: assigns}

    graph
    |> GraphProto.add(nested)
    |> move(nested_id, chain)
    |> prc_wrap(rest)
  end

  def prc_nodes2ids([]), do: []
  def prc_nodes2ids([val | rest]) when is_list(val) do
    [Enum.map(val, &(&1.id)) | prc_nodes2ids(rest)]
  end
  def prc_nodes2ids([val | rest]) do
    [val.id | prc_nodes2ids(rest)]
  end

  #-----------------------------------------------------------------------------
  # flat
  #-----------------------------------------------------------------------------
  def flat(%Cuda.Graph{nodes: [%Cuda.Graph{} = ngraph]} = graph) do
    {pins, pin_ids} = ngraph.pins
    |> Enum.reduce({[], %{}}, fn
      pin, {pins, pin_ids} ->
        id = f_find_id(graph, pin)
        {[%{pin | id: id} | pins], Map.put(pin_ids, pin.id, id)}
    end)
    links = Enum.map(ngraph.links, fn
      {{:__self__, id}, part} -> {{:__self__, pin_ids[id]}, part}
      {part, {:__self__, id}} -> {part, {:__self__, pin_ids[id]}}
      link                    -> link
    end)
    %{ngraph | id: graph.id, pins: pins, links: links}
  end
  def flat(graph), do: graph

  defp f_find_id(%{links: links}, %{id: id}) do
    links
    |> Enum.reduce_while(nil, fn
      {{:__self__, name}, {_, ^id}}, _ -> {:halt, name}
      {{_, ^id}, {:__self__, name}}, _ -> {:halt, name}
      _, _                             -> {:cont, nil}
    end)
  end
end

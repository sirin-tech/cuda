defmodule Cuda.Graph do
  @moduledoc """
  Represents evaluation graph
  """

  alias Cuda.Graph.Pin
  alias Cuda.Graph.Node

  @type id :: String.t | atom | non_neg_integer
  @type link :: {id, id}

  @type t :: %__MODULE__{
    id: id,
    module: module,
    type: Node.type,
    pins: [Pin.t],
    nodes: [Node.t],
    links: [{link, link}],
  }

  @callback __graph__(graph :: t, opts :: keyword, env :: keyword) :: t

  defstruct [:id, :module, type: :graph, pins: [], nodes: [], links: []]

  @self :__self__
  @input_pins  ~w(input consumer)a
  @output_pins ~w(output producer)a

  defmacrop error(msg) do
    quote do
      raise CompileError, description: unquote(msg)
    end
  end

  @exports [add: 3, add: 4, add: 5, link: 3]

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
      def __type__(_, _), do: :graph
    end
  end

  @doc """
  Creates new graph node
  """
  @spec new(id :: id, module :: module, opts :: keyword, env :: keyword) :: t
  def new(id, module, opts \\ [], env \\ []) do
    with {:module, module} <- Code.ensure_loaded(module) do
      graph = id
              |> Node.new(module, opts, env)
              |> Map.from_struct
      graph = struct(__MODULE__, graph)
      graph = case function_exported?(module, :__graph__, 3) do
        true -> module.__graph__(graph, opts, env)
        _    -> graph
      end
      # graph = graph |> Graph.expand()
      graph
    else
      _ -> error("Graph module #{module} could not be loaded")
    end
  end

  def add(%__MODULE__{} = graph, id, module, opts \\ [], env \\ []) do
    add(graph, Node.new(id, module, opts, env))
  end

  def add(%__MODULE__{nodes: nodes} = graph, %{id: id} = node) do
    with nil <- node(graph, id) do
      %{graph | nodes: [node | nodes]}
    else
      _ -> error("Node with id `#{id}` is already in the graph")
    end
  end

  @doc """
  Move node from source graph into destination graph, when destination graph
  is nested into source graph and moving node belongs to source graph
  """
  @spec move(source_graph :: t, destination_graph :: t, node_id :: term) :: t
  def move(srcg, _dstg, nid) do
    case node(srcg, nid) do
      nil  -> error("Node #{nid} do not belongs to #{srcg.id} graph")
      _node ->
        nil
    end
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, {dn, dp} = dst) do
    # node to node connection
    with {:src, %{} = src_node} <- {:src, node(graph, sn)},
         {:dst, %{} = dst_node} <- {:dst, node(graph, dn)} do
      src_pin = assert_pin_type(src_node, sp, @output_pins)
      dst_pin = assert_pin_type(dst_node, dp, @input_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{src, dst} | links]}
    else
      {:src, _} -> error("Source node `#{sn}` not found")
      {:dst, _} -> error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, {dn, dp} = dst) do
    # input to node connection
    with %{} = dst_node <- node(graph, dn) do
      src_pin = assert_pin_type(graph, src, @input_pins)
      dst_pin = assert_pin_type(dst_node, dp, @input_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{{@self, src}, dst} | links]}
    else
      _ -> error("Destination node `#{dn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, {sn, sp} = src, dst) do
    # node to output connection
    with %{} = src_node <- node(graph, sn) do
      src_pin = assert_pin_type(graph, dst, @output_pins)
      dst_pin = assert_pin_type(src_node, sp, @output_pins)
      assert_pin_data_type(src_pin, dst_pin)
      %{graph | links: [{src, {@self, dst}} | links]}
    else
      _ -> error("Source node `#{sn}` not found")
    end
  end

  def link(%__MODULE__{links: links} = graph, src, dst) do
    # input to output connection
    src_pin = assert_pin_type(graph, src, @input_pins)
    dst_pin = assert_pin_type(graph, dst, @output_pins)
    assert_pin_data_type(src_pin, dst_pin)
    %{graph | links: [{{@self, src}, {@self, dst}} | links]}
  end

  def dfs(graph, callback, state \\ %{})
  def dfs(%__MODULE__{} = graph, callback, state) do
    st = %{graph: graph, nodes: [], state: state}
    result = graph |> Node.get_pins(:input) |> Enum.reduce({:ok, st}, fn
      %Pin{id: id}, {:ok, st} -> dfs({:__self__, id}, callback, {:ok, st})# |> IO.inspect
      _, result -> result
    end)
    case result do
      {:error, _} = err -> err
      {action, st} -> {action, st.state}
      result       -> error("Unexpected result `#{inspect result}` returned from `dfs`")
    end
  end
  def dfs(node_spec, callback, {:ok, st}) do
    # IO.inspect(node_spec, label: :IN)

    # get node from graph by node_spec
    node = node_and_pin_by_spec(st.graph, node_spec)

    # yield callback if this is a first loopkup of node
    result = if not node_spec in st.nodes do
      with {:ok, st} <- yield(callback, :enter, node, st) do
        {:ok, %{st | nodes: [node_spec | st.nodes]}}
      end
    else
      {:ok, st}
    end
    with {:ok, st} <- result do
      result = node_spec
               |> dfs_next_spec(st.graph)
               |> Enum.reduce({:ok, st}, & dfs_reducer(&1, &2, node, callback))
      with {:ok, st} <- result do
        yield(callback, :leave, node, st)
      end
    end
  end
  def dfs(_, _, result), do: result

  defp dfs_reducer(next, {:ok, st}, node, callback) do
    next = Enum.filter(st.graph.links, fn
      {^next, dst} -> dst
      _            -> nil
    end)
    case next do
      [] ->
        error("unconnected pin detected")
      next ->
        next
        |> Enum.reduce({:ok, st}, & dfs_next_reducer(&1, &2, node, callback))
    end
  end
  defp dfs_reducer(_next, result, _node, _callback) do
    result
  end

  defp dfs_next_reducer({_, dst_spec}, {:ok, st}, node, callback) do
    case dst_spec in st.nodes do
      false ->
        dst = node_and_pin_by_spec(st.graph, dst_spec)
        with {:ok, st} <- yield(callback, :move, {node, dst}, st) do
          # recursion
          dfs(dst_spec, callback, {:ok, st})
        end
      _ ->
        {:ok, st}
    end
  end
  defp dfs_next_reducer(_next, result, _node, _callback) do
    result
  end

  def topology_sort(graph) do
    with false <- loop?(graph) do
      dfs(graph, fn
        :leave, {%{id: node}, %{id: pin}}, st -> {:ok, [{node, pin} | st]}
        _, _, st                             -> {:ok, st}
      end, [])
    else
      true  -> {:error, :loop}
      error -> error
    end
  end

  def loop?(graph) do
    result = dfs(graph, fn
      # graph input visited - reset chain
      :enter, {%__MODULE__{}, %{type: type}}, _ when type in @input_pins ->
        {:ok, MapSet.new()}
      # graph output visited - skip
      :enter, {%__MODULE__{}, _}, chain ->
        {:ok, chain}
      # node visited - add it to chain
      :enter, {%{id: node}, _}, chain ->
        {:ok, MapSet.put(chain, node)}
      # move from graph input - skip
      :move, {{%__MODULE__{}, _}, _}, chain ->
        {:ok, chain}
      # move to graph output - skip
      :move, {_, {%__MODULE__{}, _}}, chain ->
        {:ok, chain}
      # move to node - check if node already in chain
      :move, {_, {%{id: to}, _}}, chain ->
        if MapSet.member?(chain, to) do
          {:stop, true}
        else
          {:ok, chain}
        end
      # leave graph input or output - skip
      :leave, {%__MODULE__{}, _}, chain ->
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

  def expand(%__MODULE__{nodes: nodes, links: links} = graph) do
    {nodes, links} = Enum.flat_map_reduce(nodes, links, fn
      %{id: gid, type: :graph} = g, links ->
        g = g |> expand()
        nodes = g.nodes |> Enum.map(& expand_id(&1, gid))
        g_links = g.links |> Enum.map(& expand_link(&1, gid))
        links = g_links |> Enum.reduce(links, & expand_link_reducer(&1, &2, gid))
        {nodes, links}
      node, links ->
        {[node], links}
    end)
    graph = %{graph | nodes: nodes, links: links}
    if loop?(graph), do: raise error("loop detected in expanded graph")
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

  @doc """
  Finds longest node chain with specific type
  """
  @spec longest_chain(graph :: t, node_type :: Node.type) :: list
  def longest_chain(graph, node_type) do
    graph
    |> expand
    |> Node.get_pins(:input)
    |> Enum.reduce([], fn %Pin{id: id}, acc ->
      case lc_link(graph, {:__self__, id}) do
        [] ->
          acc
        links ->
          chain = links
          |> Enum.reduce([], fn
            ({_, {:__self__, _}}, acc) ->
              acc
            ({_, {node_id, _}}, acc)   ->
              node = node(graph, node_id)
              chain = longest_chain(graph, node_type, node)
              lc_max_list(chain, acc)
          end)
          lc_max_list(chain, acc)
      end
    end)
  end

  defp longest_chain(graph, type, node, current \\ [], max \\ []) do
    {current, max} = case node.type do
      ^type -> {current ++ [node], max}
      _     -> {[], lc_max_list(current, max)}
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

  def lc_out_nodes(graph, node) do
    node
    |> Node.get_pins(:output)
    |> Enum.reduce([], fn %Pin{id: id}, acc ->
      case lc_link(graph, {node.id, id}) do
        []    ->
          acc
        links ->
          nodes = links
          |> Enum.reduce([], fn
            ({_, {:__self__, _}}, acc) -> acc
            ({_, {node_id, _}}, acc)   -> [node(graph, node_id) | acc]
          end)
          nodes ++ acc
      end
    end)
    |> Enum.uniq()
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

  defp node_and_pin_by_spec(graph, {:__self__, pin}) do
    {graph, Node.get_pin(graph, pin)}
  end
  defp node_and_pin_by_spec(graph, {node, pin}) do
    node = node(graph, node)
    {node, Node.get_pin(node, pin)}
  end

  defp dfs_next_spec({:__self__, pin} = node_spec, graph) do
    case Node.get_pin(graph, pin) do
      %Pin{type: :input} -> [node_spec]
      _ -> []
    end
  end
  defp dfs_next_spec({node_id, _}, graph) do
    case node(graph, node_id) do
      nil -> []
      node -> @output_pins |> Enum.flat_map(& Node.get_pins(node, &1)) |> Enum.map(& {node_id, &1.id})
    end
  end

  defp yield(callback, action, arg, st) do
    case callback.(action, arg, st.state) do
      {action, state} -> {action, %{st | state: state}}
      _ -> error("Unexpected result returned from `dfs` callback")
    end
  end

  defp assert_pin_type(node, pin_name, types) do
    with %Pin{} = pin <- Node.get_pin(node, pin_name) do
      if not pin.type in types do
        types = types |> Enum.map(& "#{&1}") |> Enum.join(" or ")
        error("Pin `#{pin_name}` of node `#{node.id}` has a wrong type. " <>
              "The #{types} types are expected.")
      end
      pin
    else
      _ -> error("Pin `#{pin_name}` not found in node `#{node.id}`")
    end
  end

  defp assert_pin_data_type(%{data_type: t1} = p1, %{data_type: t2} = p2) do
    if t1 != t2 do
      error("The pins #{p1.id} and #{p2.id} has different types")
    end
  end

  @spec gen_id() :: id
  def gen_id do
    UUID.uuid1()
  end

  @doc """
  Returns node in the graph by its name
  """
  @spec node(graph :: t, id :: id) :: Node.t
  def node(%__MODULE__{nodes: nodes}, id) do
    nodes |> Enum.find(fn
      %{id: ^id} -> true
      _          -> false
    end)
  end
  def node(_, _), do: nil
end

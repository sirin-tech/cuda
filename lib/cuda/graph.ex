defmodule Cuda.Graph do
  @moduledoc """
  Represents evaluation graph
  """

  alias Cuda.Graph.Node

  @type id :: String.t | atom | non_neg_integer
  @type connection :: id | {id, id}

  @type t :: %__MODULE__{
    id: id,
    nodes: [Node.t],
    connections: [{id | :input, id | :output}]
  }

  @src_conn_types ~w(output producer)a
  @dst_conn_types ~w(input consumer)a

  defmacro __using__ do
  end

  defstruct [:id, nodes: [], connections: []]

  @spec gen_id() :: id
  def gen_id do
    UUID.uuid1()
  end

  @doc """
  Defines an evaluation graph.

  All operations should be specified inside do block.
  """
  defmacro graph(name \\ nil, opts \\ []) do
    {name, opts} = if is_list(name) do
      {gen_id(), name}
    else
      {name, opts}
    end
    opts = if Keyword.keyword?(opts), do: opts, else: []
    block = Keyword.get(opts, :do)
    if is_nil(block) do
      raise CompileError, description: "do block is required for graph macro"
    end
    exports = [connect: 2, connect: 3,
               input: 0,
               output: 1, output: 2,
               run: 2, run: 3]
    eval_opts = [functions: [{__MODULE__, exports} | __CALLER__.functions]]
    {graph, _bindings} = Code.eval_quoted(block, [], eval_opts)
    validate!(graph)
    graph = %{graph | id: name}
    Macro.escape(graph)
  end

  @doc """
  Returns node by its name
  """
  @spec node(graph :: t, id :: id) :: Node.t
  def node(%__MODULE__{nodes: nodes}, id) do
    nodes |> Enum.find(fn
      %Node{id: ^id} -> true
      _              -> false
    end)
  end
  def node(_, _), do: nil

  @doc """
  Creates a graph input.

  Returns newly created graph, so you can chain this function to other helpers
  like `run/3` or `connect/3`.
  """
  @spec input() :: t
  def input() do
    %__MODULE__{}
  end

  @doc """
  Creates a graph output.
  """
  @spec output(graph :: t) :: t
  @spec output(graph :: t, src :: connection) :: t
  def output(graph, src \\ nil)
  def output(%__MODULE__{connections: connections} = graph, src) do
    src = gen_src_conn(graph, src)
    %{graph | connections: connections ++ [{src, :output}]}
  end
  def output(_, _) do
    raise CompileError, description: "Invalid output/2 usage"
  end

  @doc """
  Connects two nodes.

  If one argument is specified then its used as a destination node. In this case
  last node in the graph or graph input (if there are no nodes in the graph)
  will be used as a source node.

  If two arguments is specified then first argument is a source node and second
  is a destination.

  To specify exact input or output use {node_name, pin_name} tuple.
  """
  @spec connect(graph :: t, dst :: connection) :: t
  @spec connect(graph :: t, src :: connection, dst :: connection) :: t
  def connect(graph, src, dst \\ nil)
  def connect(%__MODULE__{connections: connections} = graph, src, dst) do
    {src, dst} = if is_nil(dst), do: {nil, src}, else: {src, dst}
    src = case {src, dst} do
      # one arg form - src contains conn_id
      {_, nil} -> gen_src_conn(graph, src)
      # two args from
      # src contains {node_id, conn_id}
      {src, _} when is_tuple(src) -> gen_src_conn(graph, src)
      # src not specified - use previous node and guess connector
      {nil, _} -> gen_src_conn(graph, nil)
      # src contains node_id, connector should be guessed
      {src, _} -> gen_src_conn(graph, {src, nil})
    end
    dst = gen_dst_conn(graph, dst)
    %{graph | connections: connections ++ [{src, dst}]}
  end
  def connect(_, _, _) do
    raise CompileError, description: "Invalid connect/3 usage"
  end

  @doc """
  Adds specified module as an operation node to evaluation graph and connects
  its input to output of last node (named 'source node') in the graph.

  You can specify node name with `name` option. If not specified then it will
  be filled with a random UUID.

  If source node has several outputs and you want to get data from the specified
  output specify output name in `source` option. Firstly created operation is
  automatically connected to graph input and you should not use a `source`
  option in the frist `run/3` call. Also `source` can be specified in the
  {node_name, pin_name} form.

  If newly created node has several inputs and you want to connect source node
  to specified input, specify input name in the `input` option. In this case
  other inputs remains unconnected and you should connect it later with the
  `connect/3` function.

  You can use `source` and `input` options together.

  Returns updated graph, so you can chain several operations like this:

  ```
  input |> run(SomeOperation) |> run(OtherOperation) |> output
  ```
  """
  @spec run(graph :: t, module :: module) :: t
  @spec run(graph :: t, module :: module, options :: keyword) :: t
  def run(graph, module, opts \\ [])
  def run(%__MODULE__{} = graph, module, opts) do
    {input, opts}  = Keyword.pop(opts, :input)
    {source, opts} = Keyword.pop(opts, :source)

    node = Node.new(module, opts)
    src = gen_src_conn(graph, source)
    graph = %{graph | nodes: graph.nodes ++ [node]}
    dst = gen_dst_conn(graph, {node.id, input})

    %{graph | connections: graph.connections ++ [{src, dst}]}
  end
  def run(_, _, _) do
    raise CompileError, description: "Invalid run/3 usage"
  end

  @doc """
  Validates graph.
  """
  @spec validate!(graph :: t) :: no_return
  def validate!(%__MODULE__{connections: connections, nodes: nodes}) do
    inputs = connections |> Enum.count(fn
      {:input, _} -> true
      _           -> false
    end)
    if inputs != 1 do
      raise CompileError, description: "There are should be exectly one " <>
                                       "connection from graph input"
    end
    outputs = connections |> Enum.count(fn
      {_, :output} -> true
      _            -> false
    end)
    if outputs != 1 do
      raise CompileError, description: "There are should be exectly one " <>
                                       "connection to graph output"
    end
    available = nodes
                |> Enum.reduce([], fn node, acc ->
                  Enum.map(node.connectors, & {node.id, &1.id}) ++ acc
                end)
                |> MapSet.new
    used = connections
           |> Enum.map(&Tuple.to_list/1)
           |> List.flatten
           |> MapSet.new
           |> MapSet.delete(:input)
           |> MapSet.delete(:output)
    unconnected = MapSet.difference(available, used)
    if MapSet.size(unconnected) > 0 do
      unconnected = unconnected
                    |> Enum.into([])
                    |> Enum.map(fn {a, b} -> "#{a}.#{b}" end)
                    |> Enum.join(", ")
      raise CompileError, description: "There are unconnected connectors: "
                                       <> unconnected
    end
    true
  end
  def validate!(_) do
    raise CompileError, description: "Invalid graph"
  end

  defp available_connector(graph, node, type) do
    used = graph.connections
           |> Enum.reduce(MapSet.new, fn
             {{_, a}, {_, b}}, m -> m |> MapSet.put(a) |> MapSet.put(b)
             {{_, a}, _}, m      -> m |> MapSet.put(a)
             {_, {_, a}}, m      -> m |> MapSet.put(a)
             _, m                -> m
           end)
    node
    |> Node.connectors(type)
    |> Enum.reject(& MapSet.member?(used, &1.id))
    |> List.first
  end

  # if there are no nodes in the graph then assume :input as source connector
  defp gen_src_conn(%__MODULE__{nodes: []}, nil), do: :input
  # expicit connector specified for current_node
  defp gen_src_conn(graph, {%Node{} = node, conn_id}) do
    conn = case conn_id do
      nil     -> available_connector(graph, node, :output)
      conn_id -> node |> Node.connector(conn_id)
    end
    if is_nil(conn) do
      raise CompileError, description: "Connector #{conn_id} does not " <>
                                       "exists in node #{node.id}"
    end
    if not conn.type in @src_conn_types do
      raise CompileError, description: "Connector #{node.id}.#{conn.id} has " <>
                                       "wrong type"
    end
    {node.id, conn.id}
  end
  # reject attempts to get source in empty graph
  defp gen_src_conn(%__MODULE__{nodes: []}, _) do
    raise CompileError, description: "source option should not be used in " <>
                                     "the first operation"
  end
  # explicit node-connector
  defp gen_src_conn(graph, {node_id, conn_id}) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    gen_src_conn(graph, {node, conn_id})
  end
  # node is previous node
  defp gen_src_conn(%__MODULE__{nodes: nodes} = graph, conn_id) do
    node = List.last(nodes)
    conn = case conn_id do
      nil    -> available_connector(graph, node, :output)
      output -> node |> Node.connector(output)
    end
    if is_nil(conn) do
      raise CompileError, description: "There are no available outputs in " <>
                                       "node #{inspect node} to connect to "
    end
    if not conn.type in @src_conn_types do
      raise CompileError, description: "Connector #{node.id}.#{conn.id} has " <>
                                       "wrong type"
    end
    {node.id, conn.id}
  end
  defp gen_src_conn(_, _) do
    raise CompileError, description: "There are no available outputs to " <>
                                     "connect to"
  end

  # connector should be selected automaticaly in current node
  defp gen_dst_conn(_, {%Node{} = node, nil}) do
    conn = node |> Node.connectors(:input) |> List.first
    if is_nil(conn) do
      raise CompileError, description: "There are no available destination " <>
                                       "connectors in node #{node.id}"
    end
    {node.id, conn.id}
  end
  # explicit current_node-connector
  defp gen_dst_conn(_, {%Node{} = node, conn_id}) do
    conn = Node.connector(node, conn_id)
    if is_nil(conn) do
      raise CompileError, description: "Connector #{conn_id} does not " <>
                                       "exists in node #{node.id}"
    end
    if not conn.type in @dst_conn_types do
      raise CompileError, description: "Connector #{node.id}.#{conn_id} has " <>
                                       "wrong type"
    end
    {node.id, conn.id}
  end
  # explicit node-connector
  defp gen_dst_conn(graph, {node_id, conn_id}) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    gen_dst_conn(nil, {node, conn_id})
  end
  # node specified, connector should be selected automaticaly
  defp gen_dst_conn(graph, node_id) when not is_nil(node_id) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    conn = available_connector(graph, node, :input)
    if is_nil(conn) do
      raise CompileError, description: "There are no available destination " <>
                                       "connectors in node #{node_id}"
    end
    {node_id, conn.id}
  end
  defp gen_dst_conn(_, nil) do
    raise CompileError, description: "Invalid destination specified"
  end
end

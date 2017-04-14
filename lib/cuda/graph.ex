defmodule Cuda.Graph do
  @moduledoc """
  Represents evaluation graph
  """

  alias Cuda.Graph.Node

  @type id :: String.t | atom | non_neg_integer
  @type link :: id | {id, id}

  @type t :: %__MODULE__{
    id: id,
    nodes: [Node.t],
    links: [{id | :input, id | :output}]
  }

  @callback __graph__(graph :: t, opts :: keyword, env :: keyword) :: t

  @exports [connect: 2, connect: 3,
            #output: 1, output: 2,
            run: 2, run: 3]

  @src_pin_types ~w(output producer)a
  @dst_pin_types ~w(input consumer)a

  defmacro __using__(_opts) do
    quote do
      use Cuda.Graph.Node
      import unquote(__MODULE__), only: unquote(@exports)
      @behaviour unquote(__MODULE__)
      def __type__(_, _), do: :graph
    end
  end

  defstruct [:id, nodes: [], links: []]

  @spec gen_id() :: id
  def gen_id do
    UUID.uuid1()
  end

  @doc """
  Creates new evaluation graph
  """
  @spec new(module :: module, opts :: keyword, env :: keyword) :: t
  def new(module, opts \\ [], env \\ []) do
    graph = Node.new(module, opts, env) |> Map.from_struct
    graph = struct(__MODULE__, graph)
    graph = case function_exported?(module, :__graph__, 3) do
      true -> module.__graph__(graph, opts, env)
      _    -> graph
    end
    validate!(graph)
    graph
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
  Creates a graph output.
  """
  @spec output(graph :: t) :: t
  @spec output(graph :: t, src :: link) :: t
  def output(graph, src \\ nil)
  def output(%__MODULE__{links: links} = graph, src) do
    src = gen_src_link(graph, src)
    %{graph | links: links ++ [{src, :output}]}
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
  @spec connect(graph :: t, dst :: link) :: t
  @spec connect(graph :: t, src :: link, dst :: link) :: t
  def connect(graph, src, dst \\ nil)
  def connect(%__MODULE__{links: links} = graph, src, dst) do
    {src, dst} = if is_nil(dst), do: {nil, src}, else: {src, dst}
    src = case {src, dst} do
      # one arg form - src contains pin_id
      {_, nil} -> gen_src_link(graph, src)
      # two args from
      # src contains {node_id, pin_id}
      {src, _} when is_tuple(src) -> gen_src_link(graph, src)
      # src not specified - use previous node and guess connector
      {nil, _} -> gen_src_link(graph, nil)
      # src contains node_id, connector should be guessed
      {src, _} -> gen_src_link(graph, {src, nil})
    end
    dst = gen_dst_link(graph, dst)
    %{graph | links: links ++ [{src, dst}]}
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
    src = gen_src_link(graph, source)
    graph = %{graph | nodes: graph.nodes ++ [node]}
    dst = gen_dst_link(graph, {node.id, input})

    %{graph | links: graph.links ++ [{src, dst}]}
  end
  def run(_, _, _) do
    raise CompileError, description: "Invalid run/3 usage"
  end

  @doc """
  Validates graph.
  """
  @spec validate!(graph :: t) :: no_return
  def validate!(%__MODULE__{links: links, nodes: nodes}) do
    inputs = links |> Enum.count(fn
      {:input, _} -> true
      _           -> false
    end)
    if inputs != 1 do
      raise CompileError, description: "There are should be exectly one " <>
                                       "connection from graph input"
    end
    outputs = links |> Enum.count(fn
      {_, :output} -> true
      _            -> false
    end)
    if outputs != 1 do
      raise CompileError, description: "There are should be exectly one " <>
                                       "connection to graph output"
    end
    available = nodes
                |> Enum.reduce([], fn node, acc ->
                  Enum.map(node.pins, & {node.id, &1.id}) ++ acc
                end)
                |> MapSet.new
    used = links
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
      raise CompileError, description: "There are unconnected pins: "
                                       <> unconnected
    end
    true
  end
  def validate!(_) do
    raise CompileError, description: "Invalid graph"
  end

  defp available_pin(graph, node, type) do
    used = graph.links
           |> Enum.reduce(MapSet.new, fn
             {{_, a}, {_, b}}, m -> m |> MapSet.put(a) |> MapSet.put(b)
             {{_, a}, _}, m      -> m |> MapSet.put(a)
             {_, {_, a}}, m      -> m |> MapSet.put(a)
             _, m                -> m
           end)
    node
    |> Node.get_pins(type)
    |> Enum.reject(& MapSet.member?(used, &1.id))
    |> List.first
  end

  # if there are no nodes in the graph then assume :input as source connector
  defp gen_src_link(%__MODULE__{nodes: []}, nil), do: :input
  # expicit connector specified for current_node
  defp gen_src_link(graph, {%Node{} = node, pin_id}) do
    pin = case pin_id do
      nil    -> available_pin(graph, node, :output)
      pin_id -> node |> Node.get_pin(pin_id)
    end
    if is_nil(pin) do
      raise CompileError, description: "Pin #{pin_id} does not " <>
                                       "exists in node #{node.id}"
    end
    if not pin.type in @src_pin_types do
      raise CompileError, description: "Pin #{node.id}.#{pin.id} has " <>
                                       "wrong type"
    end
    {node.id, pin.id}
  end
  # reject attempts to get source in empty graph
  defp gen_src_link(%__MODULE__{nodes: []}, _) do
    raise CompileError, description: "source option should not be used in " <>
                                     "the first operation"
  end
  # explicit node-connector
  defp gen_src_link(graph, {node_id, pin_id}) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    gen_src_link(graph, {node, pin_id})
  end
  # node is previous node
  defp gen_src_link(%__MODULE__{nodes: nodes} = graph, pin_id) do
    node = List.last(nodes)
    pin = case pin_id do
      nil    -> available_pin(graph, node, :output)
      output -> node |> Node.get_pin(output)
    end
    if is_nil(pin) do
      raise CompileError, description: "There are no available outputs in " <>
                                       "node #{inspect node} to connect to "
    end
    if not pin.type in @src_pin_types do
      raise CompileError, description: "Connector #{node.id}.#{pin.id} has " <>
                                       "wrong type"
    end
    {node.id, pin.id}
  end
  defp gen_src_link(_, _) do
    raise CompileError, description: "There are no available outputs to " <>
                                     "connect to"
  end

  # connector should be selected automaticaly in current node
  defp gen_dst_link(_, {%Node{} = node, nil}) do
    pin = node |> Node.get_pins(:input) |> List.first
    if is_nil(pin) do
      raise CompileError, description: "There are no available destination " <>
                                       "pins in node #{node.id}"
    end
    {node.id, pin.id}
  end
  # explicit current_node-connector
  defp gen_dst_link(_, {%Node{} = node, pin_id}) do
    pin = Node.get_pin(node, pin_id)
    if is_nil(pin) do
      raise CompileError, description: "Connector #{pin_id} does not " <>
                                       "exists in node #{node.id}"
    end
    if not pin.type in @dst_pin_types do
      raise CompileError, description: "Connector #{node.id}.#{pin_id} has " <>
                                       "wrong type"
    end
    {node.id, pin.id}
  end
  # explicit node-connector
  defp gen_dst_link(graph, {node_id, pin_id}) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    gen_dst_link(nil, {node, pin_id})
  end
  # node specified, connector should be selected automaticaly
  defp gen_dst_link(graph, node_id) when not is_nil(node_id) do
    node = node(graph, node_id)
    if is_nil(node) do
      raise CompileError, description: "Node #{node_id} does not exists in " <>
                                       "graph"
    end
    pin = available_pin(graph, node, :input)
    if is_nil(pin) do
      raise CompileError, description: "There are no available destination " <>
                                       "pins in node #{node_id}"
    end
    {node_id, pin.id}
  end
  defp gen_dst_link(_, nil) do
    raise CompileError, description: "Invalid destination specified"
  end
end

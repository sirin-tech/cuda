defmodule Cuda.Test.GraphHelpers do
  @moduledoc """
  Represents helper functions for testing Cuda.Graph module
  """

  # graphics: ┌┐└┘─│▶⎡⎣⎤⎦┴┤├┬

  alias Cuda.Graph
  alias Cuda.Graph.Node
  alias Cuda.Graph.Pin

  defmodule Double do
    @moduledoc """
    Implements node with two input and output pins and specific type.
    Type is set by using a key :type in options.
    """
    use Node
    def __pins__(_) do
      [input(:input1, :i8), input(:input2, :i8),
       output(:output1, :i8), output(:output2, :i8)]
    end
    def __type__(assigns) do
      Keyword.get(assigns.options, :type, :virtual)
    end
  end

  defmodule Single do
    @moduledoc """
    Implements node with one input and output pins and specific type.
    Type is set by using a key :type in options.
    """
    use Node
    def __pins__(_) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __type__(assigns) do
      Keyword.get(assigns.options, :type, :virtual)
    end
  end

  defmodule Producer do
    @moduledoc """
    Implements node with one producer pin and specific type.
    Type is set by using a key :type in options.
    """
    use Node
    def __pins__(_) do
      [producer(:producer, :i8)]
    end
    def __type__(assigns) do
      Keyword.get(assigns.options, :type, :virtual)
    end
  end

  defmodule Custom do
    @moduledoc """
    Implements node with custom number of input and output pins and specific type.
    Type is set by using a key :type in options.
    Number of input and output pins is set by using key :io, wich takes a tuple
    {input_pins_number, output_pins_number}
    """
    use Node
    def __pins__(assigns) do
      {i, o} = Keyword.get(assigns.options, :io, {1, 1})
      inputs =  i > 0 && (for x <- 1..i, do: input(String.to_atom("input#{x}"), :i8))   || []
      outputs = o > 0 && (for x <- 1..o, do: output(String.to_atom("output#{x}"), :i8)) || []
      inputs ++ outputs
    end
    def __type__(assigns) do
      Keyword.get(assigns.options, :type, :virtual)
    end
  end

  defmodule SimpleGraph do
    @moduledoc """
    Represents a simple graph
    """
    use Graph
    def __pins__(_) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __graph__(graph) do
      graph
      |> add(:a, Single)
      |> link(:input, {:a, :input})
      |> link({:a, :output}, :output)
    end
  end

  import Graph, except: [graph: 1, graph: 2]

  @doc """
  Returns a specified graph for testing
  """
  @spec graph(atom | list) :: Graph.t
  def graph(opts \\ [])
  # [i]──▶[input (a) output]─x─▶[o]
  def graph(:unconnected) do
    graph(id: :g,
      pins: [%Pin{id: :i, type: :input, data_type: :i8},
             %Pin{id: :o, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> link(:i, {:a, :input})
  end
  # [i]──▶[input (a) output]──▶[o]
  def graph(:i1_single1_o1) do
    graph(id: :g,
          pins: [%Pin{id: :i, type: :input, data_type: :i8},
                 %Pin{id: :o, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> link(:i, {:a, :input})
    |> link({:a, :output}, :o)
  end
  # [i]─┬─▶[input (a) output]──▶[o1]
  #     └─▶[input (b) output]──▶[o2]
  def graph(:i1_single2_o2) do
    graph(id: :g,
          pins: [%Pin{id: :i, type: :input, data_type: :i8},
                 %Pin{id: :o1, type: :output, data_type: :i8},
                 %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> link(:i, {:a, :input})
    |> link(:i, {:b, :input})
    |> link({:a, :output}, :o1)
    |> link({:b, :output}, :o2)
  end
  # [i1]──▶⎡input1 (a) output1⎤──▶[o1]
  # [i2]──▶⎣input2     output2⎦──▶[o2]
  def graph(:i2_double1_o2) do
    graph(id: :g,
          pins: [%Pin{id: :i1, type: :input, data_type: :i8},
                 %Pin{id: :i2, type: :input, data_type: :i8},
                 %Pin{id: :o1, type: :output, data_type: :i8},
                 %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Double)
    |> link(:i1, {:a, :input1})
    |> link(:i2, {:a, :input2})
    |> link({:a, :output1}, :o1)
    |> link({:a, :output2}, :o2)
  end
  # [i]──▶⎡input1 (a) output1⎤──▶[o]
  #    ┌─▶⎣input2     output2⎦─┐
  #    └───────────────────────┘
  def graph(:i1_double1_o1) do
    graph(id: :g,
          pins: [%Pin{id: :i, type: :input, data_type: :i8},
                 %Pin{id: :o, type: :output, data_type: :i8}])
    |> add(:a, Double)
    |> link(:i, {:a, :input1})
    |> link({:a, :output1}, :o)
    |> link({:a, :output2}, {:a, :input2})
  end
  # [i]──▶[input (a) output]─┬──────────────────────▶[o1]
  #                          └─▶[input (b) output]──▶[o2]
  def graph(:i1_single1_single1_o2) do
    graph(id: :g,
          pins: [%Pin{id: :i, type: :input, data_type: :i8},
                 %Pin{id: :o1, type: :output, data_type: :i8},
                 %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> link(:i, {:a, :input})
    |> link({:a, :output}, :o1)
    |> link({:a, :output}, {:b, :input})
    |> link({:b, :output}, :o2)
  end
  # [i]──▶[input──▶[x-input (x-a) x-output]──▶output]──▶[o]
  def graph(:i1_graph1_o1) do
    graph(id: :g,
          pins: [%Pin{id: :i, type: :input, data_type: :i8},
                 %Pin{id: :o, type: :output, data_type: :i8}])
    |> add(:x, SimpleGraph)
    |> link(:i, {:x, :input})
    |> link({:x, :output}, :o)
  end
  # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
  #                            └─▶[input (c) output]───────────────────────▶[o2]
  def graph(:i1_single4_o2) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> add(:c, Single)
    |> add(:d, Single)
    |> link(:i1, {:a, :input})
    |> link({:a, :output}, {:b, :input})
    |> link({:a, :output}, {:c, :input})
    |> link({:b, :output}, {:d, :input})
    |> link({:d, :output}, :o1)
    |> link({:c, :output}, :o2)
  end
  # [i1]──▶[input (a) output]──┬──[input (b) output]───────────────────────▶[o1]
  #                            └─▶[input (c) output]──▶[input (d) output]──▶[o2]
  def graph(:i1_single4_o2_inverse) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> add(:c, Single)
    |> add(:d, Single)
    |> link(:i1, {:a, :input})
    |> link({:a, :output}, {:b, :input})
    |> link({:a, :output}, {:c, :input})
    |> link({:c, :output}, {:d, :input})
    |> link({:b, :output}, :o1)
    |> link({:d, :output}, :o2)
  end
  #      ┌───▶[input (a) output]───▶[input (c) output]───▶[o1]
  # [i1]─│
  #      └───▶[input (b) output]─────────────────────────▶[o2]
  def graph(:i1_single3_o2) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> add(:c, Single)
    |> link(:i1, {:a, :input})
    |> link(:i1, {:b, :input})
    |> link({:a, :output}, {:c, :input})
    |> link({:c, :output}, :o1)
    |> link({:b, :output}, :o2)
  end
  #      ┌───▶[input (a) output]─────────────────────────▶[o1]
  # [i1]─│
  #      └───▶[input (b) output]───▶[input (c) output]───▶[o2]
  def graph(:i1_single3_o2_inverse) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> add(:c, Single)
    |> link(:i1, {:a, :input})
    |> link(:i1, {:b, :input})
    |> link({:b, :output}, {:c, :input})
    |> link({:a, :output}, :o1)
    |> link({:c, :output}, :o2)
  end
  # [i1]─────▶[input (a) output]─────────────────────────▶[o1]
  #           [    (b) producer]───▶[input (c) output]───▶[o2]
  def graph(:i1_producer1_single2_o2) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Producer)
    |> add(:c, Single)
    |> link(:i1, {:a, :input})
    |> link({:b, :producer}, {:c, :input})
    |> link({:a, :output}, :o1)
    |> link({:c, :output}, :o2)
  end
  # [i1]──▶[input (a) output]──▶[input (b) output]──▶[input (c) output]──▶[o1]
  def graph(:i1_single3_o1) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8}])
    |> add(:a, Single)
    |> add(:b, Single)
    |> add(:c, Single)
    |> link(:i1, {:a, :input})
    |> link({:a, :output}, {:b, :input})
    |> link({:b, :output}, {:c, :input})
    |> link({:c, :output}, :o1)
  end
  def graph(:longest_chain_test) do
    graph(id: :graph,
          pins: [
            %Pin{id: :i1, type: :input, data_type: :i8},
            %Pin{id: :i2, type: :input, data_type: :i8},
            %Pin{id: :o1, type: :output, data_type: :i8},
            %Pin{id: :o2, type: :output, data_type: :i8}])
    |> add(:a, Custom, type: :virtual, io: {1, 2})
    |> add(:b, Single, type: :virtual)
    |> add(:c, Custom, type: :virtual, io: {1, 3})
    |> add(:d, Double, type: :gpu)
    |> add(:e, Single, type: :virtual)
    |> add(:f, Single, type: :gpu)
    |> add(:g, Single, type: :virtual)
    |> add(:h, Single, type: :gpu)
    |> add(:i, Single, type: :virtual)
    |> add(:j, Custom, type: :gpu, io: {2, 1})
    |> add(:k, Custom, type: :gpu, io: {2, 1})
    |> add(:l, Single, type: :gpu)
    |> add(:m, Single, type: :virtual)
    |> add(:n, Custom, type: :virtual, io: {2, 1})
    |> add(:o, Single, type: :gpu)
    |> link(:i1, {:a, :input1})
    |> link(:i2, {:b, :input})
    |> link({:a, :output1}, {:c, :input1})
    |> link({:a, :output2}, {:d, :input1})
    |> link({:b, :output}, {:d, :input2})
    |> link({:c, :output1}, {:e, :input})
    |> link({:c, :output2}, {:f, :input})
    |> link({:c, :output3}, {:g, :input})
    |> link({:d, :output1}, {:h, :input})
    |> link({:d, :output2}, {:i, :input})
    |> link({:e, :output}, {:j, :input1})
    |> link({:f, :output}, {:j, :input2})
    |> link({:g, :output}, {:k, :input1})
    |> link({:h, :output}, {:k, :input2})
    |> link({:i, :output}, {:l, :input})
    |> link({:j, :output1}, {:m, :input})
    |> link({:k, :output1}, {:n, :input2})
    |> link({:l, :output}, {:o, :input})
    |> link({:m, :output}, {:n, :input1})
    |> link({:n, :output1}, :o1)
    |> link({:o, :output}, :o2)
  end
  def graph(:network_test) do
    graph(id: :network,
          pins: [
            %Pin{id: :input, type: :input, data_type: :i8},
            %Pin{id: :reply, type: :input, data_type: :i8},
            %Pin{id: :output, type: :output, data_type: :i8}])
    |> add(:conv,      Single)
    |> add(:fc,        Single)
    |> add(:error,     Custom, io: {2, 1})
    |> add(:back_fc,   Custom, io: {3, 1})
    |> add(:back_conv, Custom, io: {3, 1})
    |> link(:input,                 {:back_conv, :input1})
    |> link(:input,                 {:conv,      :input})
    |> link(:reply,                 {:error,     :input2})
    |> link({:conv,      :output},  {:back_conv, :input2})
    |> link({:conv,      :output},  {:back_fc,   :input1})
    |> link({:conv,      :output},  {:fc,        :input})
    |> link({:fc,        :output},  {:back_fc,   :input2})
    |> link({:fc,        :output},  {:error,     :input1})
    |> link({:error,     :output1}, {:back_fc,   :input3})
    |> link({:back_fc,   :output1}, {:back_conv, :input3})
    |> link({:back_conv, :output1}, :output)
  end
  def graph(opts) do
    %Graph{} |> Map.merge(opts |> Enum.into(%{}))
  end

  @doc """
  Adds nested computation graph to predefined graph
  """
  @spec nested_graph(predefined_graph_name :: atom, nested_graph_name :: atom) :: Cuda.Graph.t
  def nested_graph(predefined, nested \\ :nested) do
    Code.ensure_loaded(Graph.ComputationGraph)
    nested = Graph.Factory.new(%Cuda.Graph{}, nested, Graph.ComputationGraph, [], [])
    predefined
    |> graph()
    |> Graph.GraphProto.add(nested)
  end

  @doc """
  Converts nodes to it's ids
  """
  @spec nodes2ids([Cuda.Graph.Node.t]) :: [term]
  def nodes2ids([]), do: []
  def nodes2ids([val | rest]) when is_list(val) do
    [Enum.map(val, &(&1.id)) | nodes2ids(rest)]
  end
  def nodes2ids([val | rest]) do
    [val.id | nodes2ids(rest)]
  end

  def sort_node_ids(nodes) when is_list(nodes) do
    nodes
    |> Enum.map(&sort_node_ids/1)
    |> Enum.sort(fn
      a, b when is_list(a) and is_list(b) -> length(a) >= length(b)
      a, b -> a <= b
    end)
  end
  def sort_node_ids(x), do: x

  @doc """
  Checks connection and order of connection between two nodes, before it expands the graph
  """
  @spec connected?(Cuda.Graph.t, current_node_id :: atom, next_node_id :: atom) :: boolean
  def connected?(graph, current_node_id, next_node_id) do
    graph = Cuda.Graph.Processing.expand(graph)
    graph.links
    |> Enum.any?(fn {{cnode_id, _}, {nnode_id, _}} ->
      cnid = if is_tuple(cnode_id) do
        cnode_id
        |> Tuple.to_list()
        |> List.last()
      else
        cnode_id
      end
      nnid = if is_tuple(nnode_id) do
        nnode_id
        |> Tuple.to_list()
        |> List.last()
      else
        nnode_id
      end
      cnid == current_node_id and nnid == next_node_id
    end)
  end

  def callback(action, args, state)
  def callback(action, {{node1, _pin1}, {node2, _pin2}}, state) do
    IO.puts("#{action}: #{node1.id} - #{node2.id}")
    {:ok, state ++ [{action, {node1.id, node2.id}}]}
  end
  def callback(action, {node, _pin}, state) do
    IO.puts("#{action}: #{node.id}")
    {:ok, state ++ [{action, node.id}]}
  end
  def callback(action, args, _state) do
    {action, args}
  end

  def view(%{id: id, nodes: n, links: l}) do
    IO.puts("Graph: #{id}")
    IO.puts("Nodes:")
    Enum.each(n, & IO.puts("#{&1.id}"))
    IO.puts("Links:")
    Enum.each(l, fn
      {{:__self__, gpin}, {nid, npin}} -> IO.puts("#{gpin} -> {#{nid}, #{npin}}")
      {{nid, npin}, {:__self__, gpin}} -> IO.puts("{#{nid}, #{npin}} -> #{gpin}")
      {{nid, npin}, {nid2, npin2}}     -> IO.puts("{#{nid}, #{npin}} -> {#{nid2}, #{npin2}}")
    end)
  end

  def update_node(g, _, []), do: g
  def update_node(g, node_id, [opt | rest]) do
    g = update_node(g, node_id, opt)
    update_node(g, node_id, rest)
  end
  def update_node(g, node_id, {key, value}) do
    with index when not is_nil(index) <- Enum.find_index(g.nodes, & &1.id == node_id) do
      nodes = List.update_at(g.nodes, index, fn node ->
        case key do
          :id   -> %{node | id: value}
          :type -> %{node | type: value}
          :pins -> %{node | pins: value}
          _     -> node     
        end
      end)
      %{g | nodes: nodes}
    else
      nil -> g
    end
  end
end

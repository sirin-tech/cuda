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
    def __pins__(_, _) do
      [input(:input1, :i8), input(:input2, :i8),
       output(:output1, :i8), output(:output2, :i8)]
    end
    def __type__(opts, _) do
      Keyword.get(opts, :type, :virtual)
    end
  end

  defmodule Single do
    @moduledoc """
    Implements node with one input and output pins and specific type.
    Type is set by using a key :type in options.
    """
    use Node
    def __pins__(_, _) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __type__(opts, _) do
      Keyword.get(opts, :type, :virtual)
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
    def __pins__(opts, _) do
      {i, o} = Keyword.get(opts, :io)
      inputs =  for x <- 1..i, do: input(String.to_atom("input#{x}"), :i8)
      outputs = for x <- 1..o, do: output(String.to_atom("output#{x}"), :i8)
      inputs ++ outputs
    end
    def __type__(opts, _) do
      Keyword.get(opts, :type, :virtual)
    end
  end

  defmodule SimpleGraph do
    @moduledoc """
    Represents a simple graph
    """
    use Graph
    def __pins__(_, _) do
      [input(:input, :i8), output(:output, :i8)]
    end
    def __graph__(graph, _, _) do
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
  def graph(opts) do
    %Graph{} |> Map.merge(opts |> Enum.into(%{}))
  end

  def listnode2id([]), do: []
  def listnode2id([val | rest]) when is_list(val) do
    [Enum.map(val, &(&1.id)) | listnode2id(rest)]
  end
  def listnode2id([val | rest]) do
    [val.id | listnode2id(rest)]
  end
end

defmodule GraphTest do
  use ExUnit.Case
  alias Cuda.Graph
  alias Cuda.Graph.Node
  alias Cuda.Graph.Pin

  import Cuda.Test.GraphHelpers
  import Graph, except: [graph: 1, graph: 2]

  alias Cuda.Test.GraphHelpers.Single
  alias Cuda.Test.GraphHelpers.Double

  def traverser(:move, {{%{id: node}, %{id: pin}}, {%{id: to_node}, %{id: to_pin}}}, st) do
    {:ok, st ++ [{:move, {node, pin}, {to_node, to_pin}}]}
  end
  def traverser(action, {%{id: node}, %{id: pin}}, st) do
    {:ok, st ++ [{action, {node, pin}}]}
  end

  describe "add/4" do
    test "adds nodes to graph" do
      graph = graph() |> add(Node.new(:a, Single))
      assert [%Node{id: :a}] = graph.nodes
    end

    test "rejects nodes with id that already in the graph" do
      graph = graph() |> add(Node.new(:a, Single))
      assert_raise(CompileError, fn -> graph |> add(Node.new(:a, Double)) end)
    end
  end

  describe "link/2" do
    test "links graph input to node input" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i8}])
              |> add(Node.new(:a, Single))
              |> link(:i, {:a, :input})
      assert [{{:__self__, :i}, {:a, :input}}] = graph.links
    end

    test "links node output to graph output" do
      graph = graph(pins: [%Pin{id: :o, type: :output, data_type: :i8}])
              |> add(Node.new(:a, Single))
              |> link({:a, :output}, :o)
      assert [{{:a, :output}, {:__self__, :o}}] = graph.links
    end

    test "links graph input to graph output" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i8},
                           %Pin{id: :o, type: :output, data_type: :i8}])
              |> link(:i, :o)
      assert [{{:__self__, :i}, {:__self__, :o}}] = graph.links
    end

    test "links node output to node input" do
      graph = graph()
              |> add(Node.new(:a, Single))
              |> add(Node.new(:b, Single))
              |> link({:a, :output}, {:b, :input})
      assert [{{:a, :output}, {:b, :input}}] = graph.links
    end

    test "rejects wrong pin type connection" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i8},
                           %Pin{id: :o, type: :output, data_type: :i8}])
              |> add(Node.new(:a, Single))
              |> add(Node.new(:b, Single))
      assert_raise(CompileError, fn -> graph |> link(:o, :i) end)
      assert_raise(CompileError, fn -> graph |> link(:i, {:a, :output}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :input}, {:b, :input}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :output}, {:b, :output}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :output}, :i) end)
    end

    test "rejects wrong pin data_type connection" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i16},
                           %Pin{id: :o, type: :output, data_type: :i8}])
              |> add(Node.new(:a, Single))
              |> add(Node.new(:b, Single))
      assert_raise(CompileError, fn -> graph |> link(:i, {:a, :input}) end)
    end
  end

  describe "dfs/2" do
    test "traverses graph" do
      # [i]──▶[input (a) output]──▶[o]
      {:ok, result} = dfs(graph(:i1_single1_o1), &traverser/3, [])
      assert [{:enter, {:g, :i}}, {:move, {:g, :i}, {:a, :input}},
              {:enter, {:a, :input}}, {:move, {:a, :input}, {:g, :o}},
              {:enter, {:g, :o}}, {:leave, {:g, :o}},
              {:leave, {:a, :input}},
              {:leave, {:g, :i}}] = result

      # [i]─┬─▶[input (a) output]──▶[o1]
      #     └─▶[input (b) output]──▶[o2]
      {:ok, result} = dfs(graph(:i1_single2_o2), &traverser/3, [])
      assert [{:enter, {:g, :i}}, {:move, {:g, :i}, {:b, :input}},
              {:enter, {:b, :input}}, {:move, {:b, :input}, {:g, :o2}},
              {:enter, {:g, :o2}}, {:leave, {:g, :o2}},
              {:leave, {:b, :input}},
              {:move, {:g, :i}, {:a, :input}},
              {:enter, {:a, :input}}, {:move, {:a, :input}, {:g, :o1}},
              {:enter, {:g, :o1}}, {:leave, {:g, :o1}},
              {:leave, {:a, :input}},
              {:leave, {:g, :i}}] = result

      # [i1]──▶⎡input1 (a) output1⎤──▶[o1]
      # [i2]──▶⎣input2     output2⎦──▶[o2]
      {:ok, result} = dfs(graph(:i2_double1_o2), &traverser/3, [])
      assert [{:enter, {:g, :i1}}, {:move, {:g, :i1}, {:a, :input1}},
              {:enter, {:a, :input1}}, {:move, {:a, :input1}, {:g, :o1}},
              {:enter, {:g, :o1}}, {:leave, {:g, :o1}},
              {:move, {:a, :input1}, {:g, :o2}},
              {:enter, {:g, :o2}}, {:leave, {:g, :o2}},
              {:leave, {:a, :input1}},
              {:leave, {:g, :i1}},
              {:enter, {:g, :i2}}, {:move, {:g, :i2}, {:a, :input2}},
              {:enter, {:a, :input2}}, {:leave, {:a, :input2}},
              {:leave, {:g, :i2}}] = result

      # [i]──▶⎡input1 (a) output1⎤──▶[o]
      #    ┌─▶⎣input2     output2⎦─┐
      #    └───────────────────────┘
      {:ok, result} = dfs(graph(:i1_double1_o1), &traverser/3, [])
      assert [{:enter, {:g, :i}}, {:move, {:g, :i}, {:a, :input1}},
              {:enter, {:a, :input1}}, {:move, {:a, :input1}, {:g, :o}},
              {:enter, {:g, :o}}, {:leave, {:g, :o}},
              {:move, {:a, :input1}, {:a, :input2}},
              {:enter, {:a, :input2}}, {:leave, {:a, :input2}},
              {:leave, {:a, :input1}},
              {:leave, {:g, :i}}] = result
    end

    test "raises on unconnected pins" do
      # [i]──▶[input (a) output]─x─▶[o]
      assert_raise(CompileError, fn ->
        dfs(graph(:unconnected), &traverser/3, [])
      end)
    end
  end

  describe "topology_sort/1" do
    test "sorts nodes in topology order" do
      # [i]──▶[input (a) output]─┬──────────────────────▶[o1]
      #                          └─▶[input (b) output]──▶[o2]
      graph = graph(:i1_single1_single1_o2)
      assert [%{id: :b}, %{id: :a}] = graph.nodes
      {:ok, result} = topology_sort(graph)
      assert [{:g, :i}, {:a, :input}, {:g, :o1}, {:b, :input}, {:g, :o2}] = result
    end

    test "detects loops" do
      # [i]──▶⎡input1 (a) output1⎤──▶[o]
      #    ┌─▶⎣input2     output2⎦─┐
      #    └───────────────────────┘
      assert topology_sort(graph(:i1_double1_o1)) == {:error, :loop}
    end

    test "raises on unconnected pins" do
      # [i]──▶[input (a) output]─x─▶[o]
      assert_raise(CompileError, fn -> topology_sort(graph(:unconnected)) end)
    end
  end

  describe "loop?/1" do
    test "detects loops" do
      # [i]──▶⎡input1 (a) output1⎤──▶[o]
      #    ┌─▶⎣input2     output2⎦─┐
      #    └───────────────────────┘
      assert loop?(graph(:i1_double1_o1)) == true
    end

    test "raises on unconnected pins" do
      # [i]──▶[input (a) output]─x─▶[o]
      assert_raise(CompileError, fn -> loop?(graph(:unconnected)) end)
    end

    test "returns false for non-loop graphs" do
      # [i]──▶[input (a) output]──▶[o]
      assert loop?(graph(:i1_single1_o1)) == false

      # [i1]──▶⎡input1 (a) output1⎤──▶[o1]
      # [i2]──▶⎣input2     output2⎦──▶[o2]
      assert loop?(graph(:i2_double1_o2)) == false
    end
  end

  describe "expand/1" do
    test "expands graph nodes" do
      graph = expand(graph(:i1_graph1_o1))
      assert [%{id: {:x, :a}}] = graph.nodes
      assert [{{{:x, :a}, :output}, {:__self__, :o}},
              {{:__self__, :i}, {{:x, :a}, :input}}] = graph.links
    end
  end

  describe "longest_chain/2" do
    test "finds the longest chain in graph" do

      assert :longest_chain_test
      |> graph()
      |> Graph.longest_chain(:gpu)
      |> length() == 3

      # [i1]──▶⎡input1 (a) output1⎤──▶[o1]
      # [i2]──▶⎣input2     output2⎦──▶[o2]
      assert :i2_double1_o2
      |> graph()
      |> Graph.longest_chain(:gpu)
      |> length() == 0

      # [i]──▶[input (a) output]─┬──────────────────────▶[o1]
      #                          └─▶[input (b) output]──▶[o2]
      assert :i1_single1_single1_o2
      |> graph()
      |> Graph.longest_chain(:virtual)
      |> length() == 2
    end

    test "detects loops" do
      # [i]──▶⎡input1 (a) output1⎤──▶[o]
      #    ┌─▶⎣input2     output2⎦─┐
      #    └───────────────────────┘
      assert_raise(CompileError, fn -> Graph.longest_chain(graph(:i1_double1_o1), :virtual) end)
    end

    test "raises on unconnected pins" do
      # [i]──▶[input (a) output]─x─▶[o]
      assert_raise(CompileError, fn -> Graph.longest_chain(graph(:unconnected), :virtual) end)
    end
  end
end

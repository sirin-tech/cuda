defmodule Cuda.Graph.ProcessingTest do
  use ExUnit.Case
  alias Cuda.Graph.Processing
  alias Cuda.Graph.GraphProto, as: GProto
  alias Cuda.Graph.NodeProto,  as: NProto

  import Cuda.Test.GraphHelpers
  import Processing

  # alias Cuda.Test.GraphHelpers.Single
  # alias Cuda.Test.GraphHelpers.Double

  def traverser(:move, {{%{id: node}, %{id: pin}}, {%{id: to_node}, %{id: to_pin}}}, st) do
    {:ok, st ++ [{:move, {node, pin}, {to_node, to_pin}}]}
  end
  def traverser(action, {%{id: node}, %{id: pin}}, st) do
    {:ok, st ++ [{action, {node, pin}}]}
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
    defp normalize([]), do: []
    defp normalize(result) do
      result
      |> nodes2ids()
      |> Enum.map(&(length(&1)))
      |> Enum.sort()
    end

    test "finds the longest chain in graph" do

      assert :longest_chain_test
      |> graph()
      |> longest_chain(:gpu)
      |> normalize() == [1, 1, 1, 2, 2]

      # [i1]──▶⎡input1 (a) output1⎤──▶[o1]
      # [i2]──▶⎣input2     output2⎦──▶[o2]
      assert :i2_double1_o2
      |> graph()
      |> longest_chain(:gpu)
      |> normalize() == []

      # [i]──▶[input (a) output]─┬──────────────────────▶[o1]
      #                          └─▶[input (b) output]──▶[o2]
      assert :i1_single1_single1_o2
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [2]

      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      assert :i1_single4_o2
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [1, 3]

      # [i1]──▶[input (a) output]──┬──[input (b) output]───────────────────────▶[o1]
      #                            └─▶[input (c) output]──▶[input (d) output]──▶[o2]
      assert :i1_single4_o2_inverse
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [1, 3]

      #      ┌───▶[input (a) output]───▶[input (c) output]───▶[o1]
      # [i1]─│
      #      └───▶[input (b) output]─────────────────────────▶[o2]
      assert :i1_single3_o2
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [1, 2]

      #      ┌───▶[input (a) output]─────────────────────────▶[o1]
      # [i1]─│
      #      └───▶[input (b) output]───▶[input (c) output]───▶[o2]
      assert :i1_single3_o2_inverse
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [1, 2]

      # [i1]─────▶[input (a) output]─────────────────────────▶[o1]
      #           [    (b) producer]───▶[input (c) output]───▶[o2]
      assert :i1_producer1_single2_o2
      |> graph()
      |> longest_chain(:virtual)
      |> normalize() == [1, 1]
    end

    test "detects loops" do
      # [i]──▶⎡input1 (a) output1⎤──▶[o]
      #    ┌─▶⎣input2     output2⎦─┐
      #    └───────────────────────┘
      assert_raise(CompileError, fn -> longest_chain(graph(:i1_double1_o1), :virtual) end)
    end

    test "raises on unconnected pins" do
      # [i]──▶[input (a) output]─x─▶[o]
      assert_raise(CompileError, fn -> longest_chain(graph(:unconnected), :virtual) end)
    end
  end

  describe "move/3" do
    test "node moved into empty nested graph" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      graph = move(graph, :nested, :b)
      assert graph |> GProto.node(:nested) |> GProto.node(:b) != nil
    end

    test "pins of the moved node are copied into nested graph" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      b = GProto.node(graph, :b)
      graph = move(graph, :nested, :b)
      nested = GProto.node(graph, :nested)
      assert length(b.pins) == length(nested.pins)
    end

    test "checks connection between moved node and its neighbours" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      graph = move(graph, :nested, :b)
      assert connected?(graph, :a, :b)
      assert connected?(graph, :b, :d)
    end

    test "when two nodes have shared link, the first already in nested graph, and the second moves there, shared pin of nested graph will be removed" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      graph = move(graph, :nested, :d)
      old_pin = graph |> GProto.node(:nested)
      old_pin = old_pin.pins |> Enum.find(&(&1.type == :input))
      graph = move(graph, :nested, :b)
      new_pin = graph |> GProto.node(:nested)
      new_pin = new_pin.pins |> Enum.find(&(&1.type == :input))
      assert old_pin.id != new_pin.id
      graph = nested_graph(:i1_single4_o2)
      graph = move(graph, :nested, :b)
      old_pin = graph |> GProto.node(:nested)
      old_pin = old_pin.pins |> Enum.find(&(&1.type == :output))
      graph = move(graph, :nested, :d)
      new_pin = graph |> GProto.node(:nested)
      new_pin = new_pin.pins |> Enum.find(&(&1.type == :output))
      assert old_pin.id != new_pin.id
    end

    test "when two nodes have shared link, the first already in nested graph, and the second moves there, they have direct shared link into nested graph" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      graph = graph
      |> move(:nested, :d)
      |> move(:nested, :b)
      n = GProto.node(graph, :nested)
      assert Enum.any?(n.links, fn
        {{:b, _}, {:d, _}} -> true
        _                  -> false
      end)
    end

    test "save nested graph pin when more than one node connected to it, and one of it moved to nested graph" do
      # [i1]──▶[input (a) output]──┬──[input (b) output]──▶[input (d) output]──▶[o1]
      #                            └─▶[input (c) output]───────────────────────▶[o2]
      graph = nested_graph(:i1_single4_o2)
      graph = move(graph, :nested, :a)
      pin = graph |> GProto.node(:nested)
      pin = pin.pins |> Enum.find(&(&1.type == :output))
      graph = move(graph, :nested, :b)
      n = graph |> GProto.node(:nested)
      n = n.pins
      assert Enum.any?(n, &(&1.id == pin.id))
    end

    test "general test" do
      # [i1]──▶[input (a) output]──▶[input (b) output]──▶[input (c) output]──▶[o1]
      graph = nested_graph(:i1_single3_o1)
      graph = move(graph, :nested, [:a, :b, :c])
      n = GProto.node(graph, :nested)
      assert Enum.any?(n.nodes, &(&1.id == :a))
      assert Enum.any?(n.nodes, &(&1.id == :b))
      assert Enum.any?(n.nodes, &(&1.id == :c))
    end
  end
end

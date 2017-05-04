defmodule Cuda.GraphTest do
  use ExUnit.Case
  alias Cuda.Graph
  alias Cuda.Graph.Node
  alias Cuda.Graph.Pin

  import Cuda.Test.GraphHelpers
  import Graph, except: [graph: 1, graph: 2]

  alias Cuda.Test.GraphHelpers.Single
  alias Cuda.Test.GraphHelpers.Double

  describe "add/4" do
    test "adds nodes to graph" do
      graph = graph() |> add(:a, Single)
      assert [%Node{id: :a}] = graph.nodes
    end

    test "rejects nodes with id that already in the graph" do
      graph = graph() |> add(:a, Single)
      assert_raise(CompileError, fn -> graph |> add(:a, Double) end)
    end
  end

  describe "link/2" do
    test "links graph input to node input" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i8}])
              |> add(:a, Single)
              |> link(:i, {:a, :input})
      assert [{{:__self__, :i}, {:a, :input}}] = graph.links
    end

    test "links node output to graph output" do
      graph = graph(pins: [%Pin{id: :o, type: :output, data_type: :i8}])
              |> add(:a, Single)
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
              |> add(:a, Single)
              |> add(:b, Single)
              |> link({:a, :output}, {:b, :input})
      assert [{{:a, :output}, {:b, :input}}] = graph.links
    end

    test "rejects wrong pin type connection" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i8},
                           %Pin{id: :o, type: :output, data_type: :i8}])
              |> add(:a, Single)
              |> add(:b, Single)
      assert_raise(CompileError, fn -> graph |> link(:o, :i) end)
      assert_raise(CompileError, fn -> graph |> link(:i, {:a, :output}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :input}, {:b, :input}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :output}, {:b, :output}) end)
      assert_raise(CompileError, fn -> graph |> link({:a, :output}, :i) end)
    end

    test "rejects wrong pin data_type connection" do
      graph = graph(pins: [%Pin{id: :i, type: :input, data_type: :i16},
                           %Pin{id: :o, type: :output, data_type: :i8}])
              |> add(:a, Single)
              |> add(:b, Single)
      assert_raise(CompileError, fn -> graph |> link(:i, {:a, :input}) end)
    end
  end
end

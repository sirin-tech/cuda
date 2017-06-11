defmodule Cuda.Compiler.ComputationGraphTest do
  use ExUnit.Case

  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.GPUUnit
  alias Cuda.{Graph, Graph.ComputationGraph, Graph.Factory, Graph.GPUNode, Graph.Pin}

  defmodule Node1 do
    use GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(_), do: "node1-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  defmodule Node2 do
    use GPUNode
    def __pins__(_), do: [input(:i, :i32), output(:o, :i64)]
    def __ptx__(_), do: "node2-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  defmodule Node3 do
    use GPUNode
    def __pins__(_), do: [input(:i, :i16, :inputs), output(:o, :i32, :inputs)]
    def __ptx__(_), do: "node3-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  defmodule Node4 do
    use GPUNode
    def __pins__(_), do: [input(:i, :i32, :inputs), output(:o, :i64, :inputs)]
    def __ptx__(_), do: "node4-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  describe "sources/2" do
    test "returns chained sources" do
      graph_pins = [%Pin{id: :i, type: :input, data_type: :i16},
                    %Pin{id: :o, type: :output, data_type: :i64},
                    %Pin{id: :gi, type: :input, data_type: :i16},
                    %Pin{id: :go, type: :output, data_type: :i64},
                    %Pin{id: :inputs, type: :output, alias: :inputs}]
      graph = Factory.new(%Cuda.Graph{}, :g, ComputationGraph, [], env())
              |> Map.put(:pins, graph_pins)
              |> Graph.add(:node1, Node1)
              |> Graph.add(:node2, Node2)
              |> Graph.add(:node3, Node3)
              |> Graph.add(:node4, Node4)
              |> Graph.link(:i, {:node1, :i})
              |> Graph.link({:node1, :o}, {:node2, :i})
              |> Graph.link({:node2, :o}, :o)
              |> Graph.link(:gi, {:node3, :i})
              |> Graph.link({:node3, :o}, {:node4, :i})
              |> Graph.link({:node4, :o}, :go)
      ctx = context(root: graph, vars: %{x: 10})
      {:ok, %{assigns: %{sources: sources}}} = GPUUnit.sources(graph, ctx)
      [{:ptx, n4}, {:ptx, n3}, {:ptx, n2}, {:ptx, n1}] = sources
      assert parse_ptx(n1) == ["node1-14-22"]
      assert parse_ptx(n2) == ["node2-22-14"]
      assert parse_ptx(n3) == ["node3-0-2"]
      assert parse_ptx(n4) == ["node4-2-6"]
    end
  end
end

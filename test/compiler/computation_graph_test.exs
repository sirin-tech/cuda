defmodule Cuda.Compiler.ComputationGraphTest do
  use ExUnit.Case

  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.GPUUnit

  defmodule Node1 do
    use Cuda.Graph.GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(_), do: "node1-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  defmodule Node2 do
    use Cuda.Graph.GPUNode
    def __pins__(_), do: [input(:i, :i32), output(:o, :i64)]
    def __ptx__(_), do: "node2-<%= offset(ctx, :pins, :i) %>-<%= offset(ctx, :pins, :o) %>"
  end

  describe "sources/2" do
    test "returns chained sources" do
      graph_pins = [%Cuda.Graph.Pin{id: :gi, type: :input, data_type: :i16},
                    %Cuda.Graph.Pin{id: :go, type: :output, data_type: :i64}]
      graph = Cuda.Graph.Factory.new(%Cuda.Graph{}, :g, Cuda.Graph.ComputationGraph, [], env())
              |> Map.put(:pins, graph_pins)
              |> Cuda.Graph.add(:node1, Node1)
              |> Cuda.Graph.add(:node2, Node2)
              |> Cuda.Graph.link(:gi, {:node1, :i})
              |> Cuda.Graph.link({:node1, :o}, {:node2, :i})
              |> Cuda.Graph.link({:node2, :o}, :go)
      ctx = context(root: graph, vars: %{x: 10})
      {:ok, %{assigns: %{sources: sources}}} = GPUUnit.sources(graph, ctx)
      [{:ptx, n1}, {:ptx, n2}] = sources
      assert parse_ptx(n1) == ["node2-8-0"]
      assert parse_ptx(n2) == ["node1-0-8"]
    end
  end
end

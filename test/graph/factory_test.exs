defmodule Cuda.Compiler.FactoryTest do
  use ExUnit.Case

  alias Cuda.{Graph, Graph.Factory, Graph.Node, Graph.NodeProto}

  def pins() do
    [Node.input(:i, :i16, :inputs), Node.output(:o, :i16, :outputs)]
  end

  defmodule TestNode do
    use Node
    def __assigns__(opts, _env), do: %{options: opts}
    def __pins__(assigns), do: assigns.options[:pins]
    def __type__(_assigns), do: :gpu
  end

  defmodule TestGraph do
    use Graph
    def __pins__(_opts) do
      [%{input(:i, nil) | alias: {:group, :inputs}},
       %{output(:o, nil) | alias: {:group, :outputs}}]
    end
    def __graph__(graph) do
      child_pins = [input(:i, :i16, :inputs), output(:o, :i16, :outputs)]
      graph
      |> chain(:n1, TestNode, [pins: child_pins])
      |> chain(:n2, TestNode, [pins: child_pins])
      |> close()
    end
  end

  describe "Factory.new/5" do
    test "creates node" do
      n = Factory.new(%Node{}, :n1, TestNode, pins: pins())
      assert n.__struct__() == Node
      assert n.id == :n1
      assert n.module == TestNode
    end

    test "creates graph" do
      g = Factory.new(%Graph{}, :g, TestGraph)
      assert g.__struct__() == Graph
      assert g.id == :g
      assert g.module == TestGraph
    end

    test "substitudes aliases types in pins" do
      g = Factory.new(%Graph{}, :g, TestGraph)
      assert NodeProto.pin(g, :i).data_type == %{n1: %{i: :i16}, n2: %{i: :i16}}
    end
  end
end

defmodule Cuda.Compiler.GPUNodeTest do
  use ExUnit.Case

  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.Factory

  defmodule PTXNode do
    use Cuda.Graph.GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(_), do: "PTX-<%= var(ctx, :x) %>"
  end

  defmodule CNode do
    use Cuda.Graph.GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __c__(_), do: "C-<%= var(ctx, :x) %>"
  end

  defmodule PTXCNode do
    use Cuda.Graph.GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(_), do: "PTX-<%= var(ctx, :x) %>"
    def __c__(_), do: "C-<%= var(ctx, :y) %>"
  end

  describe "sources/2" do
    test "returns ptx sources" do
      ctx = context(vars: %{x: 10})
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, PTXNode, [], env())
      {:ok, sources} = GPUUnit.sources(node, ctx)
      assert sources == [{:ptx, "PTX-10"}]
    end

    test "returns c sources" do
      ctx = context(vars: %{x: 20})
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, CNode, [], env())
      {:ok, sources} = GPUUnit.sources(node, ctx)
      assert sources == [{:c, "C-20"}]
    end

    test "returns both c and ptx sources" do
      ctx = context(vars: %{x: 10, y: 20})
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, PTXCNode, [], env())
      {:ok, sources} = GPUUnit.sources(node, ctx)
      assert sources == [{:ptx, "PTX-10"}, {:c, "C-20"}]
    end
  end
end

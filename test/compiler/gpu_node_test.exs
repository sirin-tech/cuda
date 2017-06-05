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
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, PTXNode, [], env())
      node = %{node | assigns: %{vars: %{x: 10}}}
      context = context(root: node, path: [])
      {:ok, %{assigns: %{sources: [{:ptx, ptx}]}}} = GPUUnit.sources(node, context)
      assert parse_ptx(ptx) == ["PTX-10"]
    end

    test "returns c sources" do
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, CNode, [], env())
      node = %{node | assigns: %{vars: %{x: 20}}}
      context = context(root: node, path: [])
      {:ok, %{assigns: %{sources: [{:c, c}]}}} = GPUUnit.sources(node, context)
      assert parse_c(c) == ["C-20"]
    end

    test "returns both c and ptx sources" do
      node = Factory.new(%Cuda.Graph.GPUNode{}, :node, PTXCNode, [], env())
      node = %{node | assigns: %{vars: %{x: 10, y: 20}}}
      context = context(root: node, path: [])
      {:ok, %{assigns: %{sources: [{:ptx, ptx}, {:c, c}]}}} = GPUUnit.sources(node, context)
      assert parse_ptx(ptx) == ["PTX-10"]
      assert parse_c(c) == ["C-20"]
    end
  end
end

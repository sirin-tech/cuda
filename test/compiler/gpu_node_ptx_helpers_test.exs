defmodule Cuda.Compiler.GPUNodePTXHelpersTest do
  use ExUnit.Case

  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.{NodeProto, GPUNode}
  alias Cuda.Graph.Factory
  alias Cuda.Memory

  defmodule PTXNode do
    use GPUNode
    def __pins__(_), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(node), do: Keyword.get(node.assigns.options, :ptx)
  end

  defp new_node(ptx) do
    Factory.new(%GPUNode{}, :node, PTXNode, [ptx: ptx], env())
  end

  defp gen_ptx(text, opts \\ []) do
    node = new_node(text) |> NodeProto.assign(Keyword.get(opts, :node_assigns, %{}))
    ctx = context(root: node, path: [], assigns: Keyword.get(opts, :ctx_assigns, %{}))
    {:ok, %{assigns: %{sources: [{:ptx, ptx}]}}} = GPUUnit.sources(node, ctx)
    parse_ptx(ptx)
  end

  describe "offset/2" do
    test "returns memory offset" do
      assert gen_ptx(~s{<%= offset(ctx, :pins, :i) %>}) == ["0"]
      assert gen_ptx(~s{<%= offset(ctx, :pins, :o) %>}) == ["2"]
    end
  end

  describe "shared_offset/2" do
    test "returns shared offset" do
      memory = %Memory{vars: [
        {:a, {10, %{node1: :i16, node2: :i32}}},
        {:b, {30, %{node1: :i16, node2: :i32}}}
      ]}
      assigns = %{vars: %{layer: :node1}, memory: %{shared: memory}}
      assert gen_ptx(~s{<%= shared_offset(ctx, :a) %>}, ctx_assigns: assigns) == ["10"]
      assert gen_ptx(~s{<%= shared_offset(ctx, :b) %>}, ctx_assigns: assigns) == ["30"]
      assigns = %{vars: %{layer: :node2}, memory: %{shared: memory}}
      assert gen_ptx(~s{<%= shared_offset(ctx, :a) %>}, ctx_assigns: assigns) == ["12"]
      assert gen_ptx(~s{<%= shared_offset(ctx, :b) %>}, ctx_assigns: assigns) == ["32"]
    end
  end

  describe "defkernel/2" do
    test "expands to kernel function declaration" do
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x") do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins) {}"]
    end

    test "accepts additional parameters" do
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x", a: :u8) do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins, .param .u8 a) {}"]
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x", a: u8) do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins, .param .u8 a) {}"]
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x", a: u8.ptr.local) do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins, .param .u8 .ptr .local a) {}"]
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x", a: u8.ptr.align-16) do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins, .param .u8 .ptr .align 16 a) {}"]
      ptx = gen_ptx(~s{<%= defkernel(ctx, "x", a: u8.ptr.local.align-8) do %>\n<% end %>})
      assert ptx == [".visible .entry node__x (.param .u64 .ptr pins, .param .u8 .ptr .local .align 8 a) {}"]
    end
  end
end

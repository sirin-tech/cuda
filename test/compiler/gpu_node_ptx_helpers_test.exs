defmodule Cuda.Compiler.GPUNodePTXHelpersTest do
  use ExUnit.Case

  import Cuda.Test.CudaHelpers

  alias Cuda.Compiler.GPUUnit
  alias Cuda.Graph.GPUNode
  alias Cuda.Graph.Factory

  defmodule PTXNode do
    use GPUNode
    def __pins__(_, _), do: [input(:i, :i16), output(:o, :i32)]
    def __ptx__(opts, _), do: Keyword.get(opts, :ptx)
  end

  defp new_node(ptx) do
    Factory.new(%GPUNode{}, :node, PTXNode, [ptx: ptx], env())
  end

  defp gen_ptx(text) do
    node = new_node(text)
    ctx = %{context() | assigns: %{offsets: [i: 0, o: 2]}}
    [{:ptx, ptx}] = GPUUnit.sources(node, ctx)
    parse_ptx(ptx)
  end

  describe "offset/2" do
    test "returns pin offset" do
      ptx = gen_ptx(~s{<%= offset(ctx, :i) %>})
      assert ptx == ["0"]
      ptx = gen_ptx(~s{<%= offset(ctx, :o) %>})
      assert ptx == ["2"]
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

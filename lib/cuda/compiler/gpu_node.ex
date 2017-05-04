defimpl Cuda.Compiler.GPUUnit, for: Cuda.Graph.GPUNode do
  alias Cuda.Template

  defmodule Helpers do
    def offset(ctx, name) do
      get_in(ctx.var, [:offsets, name])
    end
  end

  def sources(node, ctx) do
    opts = [context: ctx, helpers: [Helpers]]
    ptx = case node.module.__ptx__(node.options, ctx) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    ptx = ptx |> Enum.map(& Template.ptx_preprocess(&1, opts))
    c = case node.module.__c__(node.options, ctx) do
      src when is_bitstring(src) -> [src]
      src when is_list(src)      -> src
      _                          -> []
    end
    c = c |> Enum.map(& Template.c_preprocess(&1, opts))
    ptx ++ c
  end
end

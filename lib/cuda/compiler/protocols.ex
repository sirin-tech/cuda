defprotocol Cuda.Compiler.GPUUnit do
  @type source :: {:ptx | :c, String.t}
  @spec sources(item :: struct, context :: Cuda.Template.Context.t) :: [source]
  def sources(item, ctx)
end

defprotocol Cuda.Compiler.Unit do
  @spec compile(item :: struct, context :: Cuda.Template.Context.t) :: any
  def compile(item, ctx)
end

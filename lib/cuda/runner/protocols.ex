defprotocol Cuda.Runner do
  def run(node, inputs, opts \\ [])
end

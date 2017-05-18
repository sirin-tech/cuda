defprotocol Cuda.Runner do
  def load(node, opts \\ [])
  def run(node, inputs, opts \\ [])
end

defmodule Cuda.ComputationGraph do
  use Cuda.Graph
  def __type__(_opts, _env), do: :computation_graph
  def __pins__(_opts, _env), do: []
end

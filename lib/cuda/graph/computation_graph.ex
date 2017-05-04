defmodule Cuda.Graph.ComputationGraph do
  @moduledoc """
  Implements graph for internal nodes calculation
  """

  use Cuda.Graph

  def __type__(_opts, _env), do: :computation_graph
  def __pins__(_opts, _env), do: []
  def __graph__(graph, _opts, _env), do: graph
end

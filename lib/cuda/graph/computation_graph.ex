defmodule Cuda.Graph.ComputationGraph do
  @moduledoc """
  Implements graph for internal nodes calculation
  """

  use Cuda.Graph

  alias Cuda.Graph.Processing

  def __type__(_assigns), do: :computation_graph
  def __pins__(_assigns), do: []
  def __graph__(graph), do: graph

  def __run__(%{id: gid} = graph) do
    with {:ok, nodes} <- Processing.topology_sort(graph) do
      nodes = nodes
              |> Enum.map(fn {node, _pin} -> node end)
              |> Enum.reject(& &1 == gid)
      graph = %{graph | nodes: nodes}
      graph
    else
      _ -> nil
    end
  end
end

defmodule Cuda.Test.CudaHelpers do
  alias Cuda.Compiler.Context

  def env(values \\ []) do
    {:ok, env} = Cuda.Env.create()
    env |> Map.merge(values |> Enum.into(%{}))
  end

  def context(values \\ []) do
    values = values |> Enum.into(%{})
    %Context{env: env(), var: %{}} |> Map.merge(values, &context_merge/3)
  end

  defp context_merge(:env, v1, v2), do: Map.merge(v1, v2)
  defp context_merge(:var, v1, v2), do: Map.merge(v1, v2)
  defp context_merge(_, _v1, v2), do: v2
end

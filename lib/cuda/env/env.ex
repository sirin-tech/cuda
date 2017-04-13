defmodule Cuda.Env do
  @moduledoc """
  Represents environment variables manage module
  """
  alias Cuda.Env.Validation

  @type t :: %__MODULE__{
    float_size: integer(),
    int_size: integer(),
    memory_optimization: boolean()}

  defstruct [:float_size, :int_size, :memory_optimization]

  @env_var "CUDA_ENV"
  @default %{
    float_size: 32,
    int_size: 8,
    memory_optimization: true}

  @doc """
  Creates default filled env map
  """
  @spec create() :: {:ok, t}
  def create() do
    {:ok, Map.merge(%__MODULE__{}, @default)}
  end

  @doc """
  Return env map filled from :cuda config (config.exs)
  with key loaded from system env CUDA_ENV
  """
  @spec load() :: {:ok, t} | {:error, string()}
  def load() do
    case get_env() do
      nil ->
        create()
      env ->
        keys = get_keys() |> MapSet.new()
        {:ok, init} = create()
        fill_in(env, keys, init)
    end
  end

  @doc """
  Merge env map with keyword list opts
  """
  @spec merge(t, [{atom(), any()}]) :: {:ok, t} | {:error, string()}
  def merge(env, opts) do
    keys = get_keys() |> MapSet.new()
    fill_in(opts, keys, env)
  end

  @doc """
  Return default env values map
  """
  @spec get_default() :: map()
  def get_default(), do: @default

  defp get_keys() do
    %__MODULE__{}
    |> Map.keys()
    |> Enum.filter(&(&1 != :__struct__))
  end

  defp get_env() do
    case System.get_env(@env_var) do
      nil ->
        nil
      val ->
        val = String.to_atom(val)
        Application.get_env(:cuda, val)
    end
  end

  defp fill_in([], _, env), do: {:ok, env}
  defp fill_in([{key, value} | rest], keys, env) do
    case Validation.validate(key, value) do
      {:ok, value} ->
        if MapSet.member?(keys, key) do
          fill_in(rest, keys, Map.put(env, key, value))
        else
          {:error, "unexpected value name in config"}
        end
      error        ->
        error
    end
  end
end

defmodule Cuda.Env do
  @moduledoc """
  Represents environment variables manage module
  """
  alias Cuda.Env.Validation

  @type optimize :: :memory | :speed | :adaptive | :none
  @type float_size :: 16 | 32 | 64
  @type int_size :: 8 | 16 | 32 | 64

  @type t :: %__MODULE__{
    float_size: float_size,
    int_size: int_size,
    optimize: optimize,
    gpu_info: keyword
  }

  defstruct [float_size: 4, int_size: 1, optimize: :none, gpu_info: []]

  @env_var "CUDA_ENV"
  @default %{
    float_size: 4,
    int_size: 1,
    optimize: :none}

  @doc """
  Creates default filled env map
  """
  @spec create() :: {:ok, t}
  def create() do
    {:ok, Map.merge(%__MODULE__{}, @default)}
  end

  @doc """
  Returns env map filled from :cuda config (config.exs)
  with key loaded from system env CUDA_ENV
  """
  @spec load() :: {:ok, t} | Cuda.error_tuple
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
  @spec merge(t, [keyword]) :: {:ok, t} | Cuda.error_tuple
  def merge(env, opts) do
    keys = get_keys() |> MapSet.new()
    fill_in(opts, keys, env)
  end

  @doc """
  Returns default env values map
  """
  @spec get_default() :: map
  def get_default(), do: @default

  def f(%__MODULE__{float_size: size}) when is_integer(size) do
    "f#{size * 8}"
  end
  def f(_), do: "f32"

  defp get_keys() do
    %__MODULE__{}
    |> Map.from_struct()
    |> Map.keys()
  end

  defp get_env() do
    with val when not is_nil(val) <- System.get_env(@env_var) do
      val = String.to_atom(val)
      Application.get_env(:cuda, val)
    end
  end

  defp fill_in([], _, env), do: {:ok, env}
  defp fill_in([{key, value} | rest], keys, env) do
    with {:ok, value} <- Validation.validate(key, value) do
      if MapSet.member?(keys, key) do
        fill_in(rest, keys, Map.put(env, key, value))
      else
        {:error, "unexpected value name in config"}
      end
    end
  end
end

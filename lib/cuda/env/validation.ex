defmodule Cuda.Env.Validation do
  @moduledoc """
  Represents module for validate environment values
  """

  @optimize_values ~w(memory speed adaptive none)a
  @int_size_values [8, 16, 32]
  @float_size_values [16, 32]

  @doc """
  Validates environment variables
  """
  @spec validate(atom, any) :: {:ok, any} | Cuda.error_tuple
  def validate(:optimize, value) do
    if Enum.member?(@optimize_values, value) do
      {:ok, value}
    else
      values = @optimize_values |> Enum.map(&Atom.to_string/1) |> Enum.join(", ")
      validate_error("wrong memory_optimization value, permitted values: #{values}")
    end
  end
  def validate(:int_size, value) do
    if Enum.member?(@int_size_values, value) do
      {:ok, value}
    else
      values = @int_size_values |> Enum.map(&Integer.to_string/1) |> Enum.join(", ")
      validate_error("wrong int_size value, permitted values: #{values}")
    end
  end
  def validate(:float_size, value) do
    if Enum.member?(@float_size_values, value) do
      {:ok, value}
    else
      values = @float_size_values |> Enum.map(&Integer.to_string/1) |> Enum.join(", ")
      validate_error("wrong float_size value, permitted values: #{values}")
    end
  end
  def validate(_, value), do: {:ok, value}

  defp validate_error(error_message), do: {:error, error_message}
end

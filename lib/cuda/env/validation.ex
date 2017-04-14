defmodule Cuda.Env.Validation do
@moduledoc """
Represents module for validate environment values
"""

  @doc """
  Validates environment variables
  """
  @spec validate(atom, any) :: {:ok, any} | Cuda.error_tuple
  def validate(:optimize, value) do
    if Enum.member?([:memory, :speed, :adaptive, :none], value) do
      {:ok, value}
    else
      validate_error("wrong memory_optimization value, permitted values: true, false")
    end
  end
  def validate(:int_size, value) do
    if Enum.member?([8, 16, 32], value) do
      {:ok, value}
    else
      validate_error("wrong int_size value, permitted values: 8, 16, 32")
    end
  end
  def validate(:float_size, value) do
    if Enum.member?([16, 32], value) do
      {:ok, value}
    else
      validate_error("wrong float_size value, permitted values: 16, 32")
    end
  end
  def validate(_, value), do: {:ok, value}

  defp validate_error(error_message), do: {:error, error_message}
end

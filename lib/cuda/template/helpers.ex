defmodule Cuda.Template.Helpers do
  @moduledoc """
  Represents set of helper functions for EEx templates
  """

  @doc """
  Returns Cuda environment variable value
  """
  @spec env(context :: Cuda.Template.Context.t, variable_name :: String.t | atom | number) :: any
  def env(ctx, var_name) do
    Map.get(ctx.env, var_name)
  end

  @doc """
  Returns context variable value
  """
  @spec var(context :: Cuda.Template.Context.t, variable_name :: String.t | atom | number) :: any
  def var(ctx, var_name) do
    get_in(ctx.vars, [var_name])
  end
end

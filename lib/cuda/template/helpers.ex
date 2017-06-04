defmodule Cuda.Template.Helpers do
  @moduledoc """
  Represents set of helper functions for EEx templates
  """

  alias Cuda.Compiler.Context

  @doc """
  Returns Cuda environment variable value
  """
  @spec env(context :: Context.t, variable_name :: String.t | atom | number) :: any
  def env(ctx, var_name) do
    Map.get(ctx.env, var_name)
  end

  @doc """
  Returns context variable value
  """
  @spec var(context :: Context.t, variable_name :: String.t | atom | number) :: any
  def var(ctx, var_name) do
    with nil <- Context.find_assign(ctx, [:vars, var_name]) do
      get_in(ctx.assigns, [:vars, var_name])
    end# |> IO.inspect(label: "VAR #{inspect var_name}")
  end

  defmacro var(var_name) do
    quote do
      var(var!(ctx), unquote(var_name))
    end
  end

  defmacro env(var_name) do
    quote do
      env(var!(ctx), unquote(var_name))
    end
  end
end

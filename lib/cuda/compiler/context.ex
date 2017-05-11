defmodule Cuda.Compiler.Context do
  @moduledoc """
  Compilation context
  """

  @type t :: %__MODULE__{
    env: Cuda.Env.t,
    vars: map,
    assigns: map,
    node: map}

  defstruct [:env, vars: %{}, assigns: %{}, node: %{}]
end

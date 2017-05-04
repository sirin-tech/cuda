defmodule Cuda.Compiler.Context do
  @moduledoc """
  Compilation context
  """

  @type t :: %__MODULE__{
    env: Cuda.Env.t,
    var: map}

  defstruct [:env, :var]
end

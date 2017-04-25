defmodule Cuda.Template.Context do
  @moduledoc """
  Structure fot Cuda.Template module
  """

  @type t :: %__MODULE__{
    env: Cuda.Env.t,
    var: map}

  defstruct [:env, :var]
end

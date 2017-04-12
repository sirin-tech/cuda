defmodule Cuda.App do
  @moduledoc false

  use Application

  def start(_type, _args) do
    import Supervisor.Spec

    children = [
      worker(Cuda, [], restart: :temporary)
    ]
    Supervisor.start_link(children, strategy: :simple_one_for_one, name: __MODULE__)
  end

  def start_driver(opts \\ []) do
    Supervisor.start_child(__MODULE__, opts)
  end
end

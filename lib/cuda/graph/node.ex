defmodule Cuda.Graph.Node do
  @moduledoc """
  Represents evaluation graph node.
  """

  alias Cuda.Graph
  alias Cuda.Graph.Connector

  @type t :: %__MODULE__{
    id: Graph.id,
    module: module,
    type: :gpu | :host | :virtual,
    connectors: [Connector.t]
  }
  @types ~w(gpu host virtual)a
  @reserved_names ~w(input output)a

  defstruct [:id, :module, :type, connectors: []]

  @doc """
  Creates a new evaluation node
  """
  @spec new(module :: module, options :: keyword, env :: keyword) :: t
  def new(module, opts \\ [], env \\ []) do
    id = Keyword.get(opts, :name, Graph.gen_id())
    if id in @reserved_names do
      raise CompileError, description: "Reserved node name '#{id}' used"
    end

    type = case function_exported?(module, :type, 2) do
      true -> module.type(opts, env)
      _    -> :virtual
    end
    if not type in @types do
      raise CompileError, description: "Unsupported type: #{inspect type}"
    end

    connectors = case function_exported?(module, :connectors, 2) do
      true -> module.connectors(opts, env)
      _    -> []
    end
    if not is_list(connectors) or not Enum.all?(connectors, &valid_connector/1) do
      raise CompileError, description: "Invalid connector list supploed"
    end

    %__MODULE__{
      id: id,
      module: module,
      type: type,
      connectors: connectors
    }
  end

  @doc """
  Returns connector by its id
  """
  @spec connector(node:: t, id: Graph.id) :: Connector.t | nil
  def connector(%__MODULE__{connectors: connectors}, id) do
    connectors |> Enum.find(fn
      %Connector{id: ^id} -> true
      _                   -> false
    end)
  end
  def connector(_, _), do: nil

  @doc """
  Returns a list of connectors of specified type
  """
  @spec connectors(node :: t, type :: Connector.type) :: Connector.t | nil
  def connectors(%__MODULE__{connectors: connectors}, type) do
    connectors |> Enum.filter(fn
      %Connector{type: ^type} -> true
      _                       -> false
    end)
  end
  def connectors(_, _), do: nil

  defp valid_connector(%Connector{}), do: true
  defp valid_connector(_), do: false
end

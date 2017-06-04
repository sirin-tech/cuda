defmodule Cuda.Compiler.Context do
  @moduledoc """
  Compilation context
  """

  alias Cuda.Graph.GraphProto

  @type t :: %__MODULE__{
    env: Cuda.Env.t,
    assigns: map,
    root: Cuda.Graph.t | Cuda.Node.t,
    path: [Cuda.Graph.id]
  }

  defstruct [:env, :root, assigns: %{}, path: []]

  def new(opts) do
    struct(__MODULE__, Enum.into(opts, []))
  end

  def assign(%__MODULE__{assigns: assigns} = ctx, values) do
    %{ctx | assigns: Map.merge(assigns, Enum.into(values, %{}))}
  end
  def assign(%__MODULE__{assigns: assigns} = ctx, key, value) do
    %{ctx | assigns: Map.put(assigns, key, value)}
  end

  defp expanded_path(path) do
    path
    |> Enum.flat_map(fn
      id when is_tuple(id) -> id |> Tuple.to_list |> Enum.reverse
      id                   -> [id]
    end)
  end

  defp find_in_expanded(_, _, []), do: nil
  defp find_in_expanded(assigns, path, [_ | rest] = ctx_path) do
    expanded = ctx_path |> Enum.reverse |> Enum.intersperse(:expanded)
    expanded = [:expanded | expanded] ++ path
    with nil <- get_in(assigns, expanded) do
      find_in_expanded(assigns, path, rest)
    end
  end

  def find_assign(ctx, path, ctx_path \\ nil, callback \\ fn _ -> true end)
  def find_assign(ctx, path, nil, callback) do
    find_assign(ctx, path, ctx.path, callback)
  end
  def find_assign(%{root: nil}, _, _, _), do: nil
  def find_assign(ctx, path, [], callback) do
    value = with nil <- get_in(ctx.root.assigns, [:expanded | path]) do
      get_in(ctx.root.assigns, path)
    end
    if not is_nil(value) and callback.(value), do: value, else: nil
  end
  def find_assign(ctx, path, [_ | rest] = ctx_path, callback) do
    with nil <- find_in_expanded(ctx.root.assigns, path, expanded_path(ctx_path)) do
      with %{} = child <- node(ctx, ctx_path) do
        with nil <- find_in_expanded(child.assigns, path, expanded_path(rest)),
             nil <- get_in(child.assigns, path) do
          find_assign(ctx, path, rest, callback)
        else
          value ->
            if callback.(value) do
              value
            else
              find_assign(ctx, path, rest, callback)
            end
        end
      else
        _ -> find_assign(ctx, path, rest, callback)
      end
    end
  end
  def find_assign(_, _, _, _), do: nil

  def for_node(%__MODULE__{root: nil} = ctx, child) do
    %{ctx | root: child, path: []}
  end
  def for_node(%__MODULE__{path: path} = ctx, %{id: id}) do
    {_, with_id} = Enum.split_while(path, & &1 != id)
    path = if with_id == [], do: [id | path], else: with_id
    %{ctx | path: path}
  end

  def node(ctx, node \\ nil)
  def node(%__MODULE__{root: root, path: []}, nil), do: root
  def node(%__MODULE__{path: path} = ctx, nil), do: node(ctx, path)
  def node(%__MODULE__{root: nil}, _), do: nil
  def node(%__MODULE__{root: root}, []) do
    root
  end
  def node(%__MODULE__{root: root}, path) do
    GraphProto.node(root, Enum.reverse(path))
  end
  def node(_, _) do
    nil
  end

  def replace_current(%__MODULE__{path: []} = ctx, node) do
    %{ctx | root: node}
  end
  def replace_current(%__MODULE__{root: root, path: [_ | rest] = id} = ctx, node) do
    %{ctx | path: [node.id | rest], root: GraphProto.replace(root, id |> Enum.reverse, node)}
  end
end

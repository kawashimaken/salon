import React from "react";
import AddTodo from "./components/AddTodo";
import TodoList from "./components/TodoList";
import Joke from "./components/Joke";
import RouterDemo from "./components/RouterDemo";
import VisibilityFilters from "./components/VisibilityFilters";
import "./styles.css";
import { Card, Elevation } from "@blueprintjs/core";


export default function TodoApp() {
  return (
    <div style={{ margin: 100 }} className="todo-app">
      <Card interactive={true} elevation={Elevation.FOUR}>
        <h1>Todo List（React+Redux+Blueprintデモ）</h1>
        <RouterDemo />
        <Joke />
        <AddTodo />
        <TodoList />
        <VisibilityFilters />
      </Card>
    </div>
  );
}

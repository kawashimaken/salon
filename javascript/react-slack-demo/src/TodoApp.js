import React from "react";
import AddTodo from "./components/AddTodo";
import TodoList from "./components/TodoList";
import Joke from "./components/Joke";
import RouterDemo from "./components/RouterDemo";
import SlackDemo from "./components/Slack";
import VisibilityFilters from "./components/VisibilityFilters";
import "./styles.css";
import { Card, Elevation } from "@blueprintjs/core";


export default function TodoApp() {
  return (
    <div style={{ margin: 100 }} className="todo-app">
      <Card interactive={true} elevation={Elevation.FOUR}>
        <h1>React+Slackデモ</h1>
        <SlackDemo />
        <RouterDemo />
        <Joke />
        <AddTodo />
        <TodoList />
        <VisibilityFilters />
      </Card>
    </div>
  );
}

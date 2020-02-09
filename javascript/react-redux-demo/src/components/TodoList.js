import React from "react";
import { connect } from "react-redux";
import Todo from "./Todo";
// import { getTodos } from "../redux/selectors";
import { getTodosByVisibilityFilter } from "../redux/selectors";
import { VISIBILITY_FILTERS } from "../constants";
import { Button,FormGroup, InputGroup,Card, Elevation,ControlGroup} from "@blueprintjs/core";


const TodoList = ({ todos }) => (
  <div style={{margin:10}}>
    <Card interactive={true} elevation={Elevation.FOUR}>
    <h5><a href="#">今のToDoリスト</a></h5>
      <ul className="todo-list">
      <div style={{margin:3}} className="bp3-callout bp3-intent-warning">

          {todos && todos.length
            ? todos.map((todo, index) => {
                return (<div><h4 class="todo-list" key={`todo-${todo.id}`} todo={todo}></h4>
                <Todo key={`todo-${todo.id}`} todo={todo} /></div>);
              })
            : "まだToDoがありません"}
        </div>
      </ul>
    </Card>

  </div>

);

// const mapStateToProps = state => {
//   const { byIds, allIds } = state.todos || {};
//   const todos =
//     allIds && state.todos.allIds.length
//       ? allIds.map(id => (byIds ? { ...byIds[id], id } : null))
//       : null;
//   return { todos };
// };

const mapStateToProps = state => {
  const { visibilityFilter } = state;
  const todos = getTodosByVisibilityFilter(state, visibilityFilter);
  return { todos };
  //   const allTodos = getTodos(state);
  //   return {
  //     todos:
  //       visibilityFilter === VISIBILITY_FILTERS.ALL
  //         ? allTodos
  //         : visibilityFilter === VISIBILITY_FILTERS.COMPLETED
  //           ? allTodos.filter(todo => todo.completed)
  //           : allTodos.filter(todo => !todo.completed)
  //   };
};
// export default TodoList;
export default connect(mapStateToProps)(TodoList);

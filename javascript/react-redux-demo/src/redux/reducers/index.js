import { combineReducers } from "redux";
import visibilityFilter from "./visibilityFilter";
import todos from "./todos";
import joke from "./joke"

export default combineReducers({ todos, joke, visibilityFilter });

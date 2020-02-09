import { ADD_TODO, TOGGLE_TODO, SET_FILTER,JOKE , JOKE_FETCH_FAIL} from "./actionTypes";
import axios from 'axios'

let nextTodoId = 0;

export const addTodo = content => ({
  type: ADD_TODO,
  payload: {
    id: ++nextTodoId,
    content
  }
});

export const toggleTodo = id => ({
  type: TOGGLE_TODO,
  payload: { id }
});



export const setFilter = filter => ({ type: SET_FILTER, payload: { filter } });

//
export const getJoke = () => {
  return (dispatch) => {
    return axios.get('https://official-joke-api.appspot.com/random_joke')
      .then(res =>{
        console.log(res.data)
        dispatch({
          type:JOKE,
          payload:res
        })
      }
      ).catch(err => 
        dispatch({
          type:JOKE_FETCH_FAIL
        })
      )
  }
}

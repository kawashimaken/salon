import { JOKE, JOKE_FETCH_FAIL } from "../actionTypes";

const initialState = {
  joke:''
};

export default function(state = initialState, action) {
  switch (action.type) {
    case JOKE: {
      console.log('in reducer',action.payload.data.setup)
      return {
        ...state,
        joke:action.payload.data.setup
      };
    }
    default:
      return state;
  }
}



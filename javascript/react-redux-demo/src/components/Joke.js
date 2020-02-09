import React from "react";
import { connect } from "react-redux";
import { getJoke} from "../redux/actions";
//
import { Button,Card, Elevation} from "@blueprintjs/core";

class Jokes extends React.Component {
  constructor(props) {
    super(props);
    this.state = { joke:""};
  }

  handleJoke=()=>{
    this.props.getJoke();
    console.log(this.props.joke)
  }

  render() {
    return (
      <div style={{margin:10}}>
        <Card interactive={true} elevation={Elevation.FOUR}>
        <Button intent="primary" text="Get A Joke（Fetch from an API）" onClick={this.props.getJoke} />
        <div>非同期処理の結果（APIからとってきたデータ）を下に表示する</div>
        <div className="bp3-callout bp3-intent-success">{this.props.joke}</div>
        
        </Card>
      </div>
    );
  }
}
const mapStateToProps = store => {
  //変数

  return { 
    joke:store.joke.joke 
  };

};
// export default connect(
//   null,
//   { getJoke }
// )(Jokes);
// // export default AddTodo;

function matchDispachToProp(dispatch) {
  //操作,「this.props.joke」で使う
  return {
    getJoke: () => dispatch(getJoke())
  };
}

export default connect(mapStateToProps,matchDispachToProp)(Jokes);

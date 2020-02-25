import React from "react";
import { connect } from "react-redux";
import { addTodo } from "../redux/actions";
//
import { Button,FormGroup, InputGroup,Card, Elevation,ControlGroup} from "@blueprintjs/core";

class AddTodo extends React.Component {
  constructor(props) {
    super(props);
    this.state = { input: "" };
  }

  updateInput = input => {
    this.setState({ input });
  };

  handleAddTodo = () => {
    this.props.addTodo(this.state.input);
    this.setState({ input: "" });
  };

  render() {
    return (
      <div style={{margin:10}}>
        <Card interactive={true} elevation={Elevation.FOUR}>
        <h5><a href="#">ToDoの追加</a></h5>
        <FormGroup
            helperText="ボタンをクリックしたら追加しますよ..."
            label="ToDoを書き込んでください"
            labelFor="text-input"
            labelInfo="(空白じゃないのがいいな)"
            >
              <ControlGroup fill={true} vertical={false}>
              <InputGroup
                onChange={e => this.updateInput(e.target.value)}
                value={this.state.input}
              />
              <Button intent="success" text="ToDoを追加する" onClick={this.handleAddTodo} />
              </ControlGroup>
            </FormGroup>
        </Card>
      </div>
    );
  }
}

export default connect(
  null,
  { addTodo }
)(AddTodo);
// export default AddTodo;

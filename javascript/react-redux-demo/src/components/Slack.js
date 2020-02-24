import React from "react";
import { connect } from "react-redux";
// import { getJoke } from "../redux/actions";
const { WebClient } = require('@slack/web-api');
// secret_example.jsのように、SLACK_TOKENなどを設定しておいてください
// 用意したファイルを「secret.js」に(rename)してください
import { SLACK_TOKEN } from './secret'
// UI
import { Button, Card, Elevation } from "@blueprintjs/core";

// Read a token from the environment variables
const token = SLACK_TOKEN;

// Initialize
const web = new WebClient(token);


// Jokeというコンポーネントを作る
class Slack extends React.Component {
    constructor(props) {
        super(props);
        //初期のstateにjokeを作って、空の文字列に設定する
        this.state = { slack: "" };
    }
    //JokeをAPIサーバからとってくる処理
    //実際、中身はRedux のAction（getJoke）に接続する
    handleSlack = () => {
        // this.props.getJoke();
        // console.log(this.props.joke)
    }
    //UIをレンダリングする
    render() {
        return (
            <div style={{ margin: 10 }}>
                <Card interactive={true} elevation={Elevation.FOUR}>
                    {/* <Button intent="primary" text="Get A Joke（Fetch from an API）" onClick={this.props.getJoke} /> */}
                    <div>Slack Web API Demo</div>
                    {/* <div className="bp3-callout bp3-intent-success">{this.props.joke}</div> */}
                </Card>
            </div>
        );
    }
}

//Reduxのstateをこのコンポーネントのpropsにするよ
const mapStateToProps = store => {
    //変数

    return {
        //左は、ここのstate.joke
        //右は、Reduxのstoreからくるstate, jokeの配下のjoke
        //reducer/joke.jsを参照してください
        //joke: store.joke.joke
    };

};

//このコンポーネントのActionとReduxのActionにマッピングする
function matchDispachToProp(dispatch) {
    //操作,「this.props.joke」で使う
    return {
        //左は、このコンポーネントで使う関数
        //右は、Reduxで定義したAction（要は関数）
        //こうすることで、このコンポーネントでReduxのActionが使えるようになる
        //Actionは全部Reduxにまとめて、管理する
        //各コンポーネントは、こういう形で接続して、ReduxのActionを呼び出す
        //Actionをそれぞれのコンポーネントに分散するのではなく、ReduxのActionで一元管理する
        //ReduxのActionを見れば、アプリケーションの全貌が把握できる
        //getJoke: () => dispatch(getJoke())
    };
}

//Jokeというコンポーネントを「外」から見えるようにする
//かつ、上のmapStateToPropsとmatchDispachToPropをリンクさせて、Reduxの仕組みを有効にする
//mapStateToPropsとmatchDispachToPropは下記の順番で、connectすれば、別の名前でも良い
export default connect(mapStateToProps, matchDispachToProp)(Slack);

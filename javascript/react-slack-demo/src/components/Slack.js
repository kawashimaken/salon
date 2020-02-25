import React from "react";

// secret_example.jsのように、SLACK_TOKENなどを設定しておいてください
// 用意したファイルを「secret.js」に(rename)してください
import { SLACK_TOKEN, SLACK_TOKEN_BOT_USER } from './secret'
// UI
import { Button, Card, Elevation } from "@blueprintjs/core";

//
const { WebClient } = require('@slack/web-api');
// An access token (from your Slack app or custom integration - xoxp, xoxb)
const bot_user_token = SLACK_TOKEN_BOT_USER;

const web = new WebClient(bot_user_token);


// Slackというコンポーネントを作る
class Slack extends React.Component {
    constructor(props) {
        super(props);
        //
        this.state = { slack_user_info: "" };
    }
    //
    //
    handleSlack = async () => {
        // The current date
        const currentTime = new Date().toTimeString();
        try {
            // Use the `chat.postMessage` method to send a message from this app
            await web.chat.postMessage({
                channel: '#slack_bot',
                text: `川島のプログラムからテスト！The current time is ${currentTime}`,
            });
        } catch (error) {
            console.log(error);
        }

        console.log('Message posted!');
    }

    handleSlackUser = async () => {
        // The current date
        const currentTime = new Date().toTimeString();
        try {
            // Use the `chat.postMessage` method to send a message from this app
            var result = await web.users.identity({
                token: SLACK_TOKEN
            });
            this.setState({ slack_user_info: result })
            console.log(result)
        } catch (error) {
            console.log(error);
        }

        console.log('Message posted!');
    }
    //UIをレンダリングする
    render() {
        return (
            <div style={{ margin: 10 }}>
                <Card interactive={true} elevation={Elevation.FOUR}>
                    <Button intent="primary" text="Go Slack（hit slack API）" onClick={this.handleSlack} />
                    <Button intent="primary" text="Slack User" onClick={this.handleSlackUser} />
                    <div>Slack Web API Demo</div>
                    {/* <div className="bp3-callout bp3-intent-success">{this.state.slack_user_info.user.id}</div> */}
                </Card>
            </div>
        );
    }
}



export default Slack;


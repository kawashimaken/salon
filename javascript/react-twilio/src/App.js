import React from 'react';
import './App.css';
import firebase from 'firebase/app'; //必須
import 'firebase/firestore'; //必要なモジュールごとにimport
//
// secret_example.jsのように、twilioのAPIなどを設定しておいてください
// 用意したファイルを「secret.js」にしてください
import { firebaseConfig, ACCOUNT_SID, API_KEY_SID, API_KEY_SECRET } from './secret'

const {
  connect
} = require('twilio-video');

/*

*/
class App extends React.Component {

  constructor(props) {
    super(props);
    // ここで this.setState() を呼び出さないでください
    this.state = { date_string: '' };
    //インスタンスの初期化
    firebase.initializeApp(firebaseConfig);
    this.state.db = firebase.firestore();

  }

  componentDidMount() {
    this.get_ready()
  }

  get_ready() {
    var AccessToken = require('twilio').jwt.AccessToken;
    var VideoGrant = AccessToken.VideoGrant;

    // Create an Access Token
    var accessToken = new AccessToken(
      ACCOUNT_SID,
      API_KEY_SID,
      API_KEY_SECRET
    );

    // Set the Identity of this token
    accessToken.identity = 'example-user';

    // Grant access to Video
    var grant = new VideoGrant();
    grant.room = 'DemoRoom';
    accessToken.addGrant(grant);

    // Serialize the token as a JWT
    var jwt = accessToken.toJwt();
    console.log(jwt);
    //

    connect(jwt, {
      name: 'DemoRoom'
    }).then(room => {
      console.log(`Successfully joined a Room: ${room}`);
      //
      room.on('participantConnected', participant => {
        console.log(`A remote Participant connected: ${participant}`);
      });
    }, error => {
      console.error(`Unable to connect to Room: ${error.message}`);
    });

    //
    // Set the Identity of this token
    accessToken.identity = 'another-user';

    // Grant access to Video
    var grant = new VideoGrant();
    grant.room = 'DemoRoom';
    accessToken.addGrant(grant);

    // Serialize the token as a JWT
    var another_jwt = accessToken.toJwt();
    console.log(another_jwt);

    //
    connect(another_jwt, {
      name: 'DemoRoom'
    }).then(room => {
      console.log(`Another User Successfully joined a Room: ${room}`);
      //
      room.on('participantConnected', participant => {
        console.log(`Another User said: A remote Participant connected: ${participant}`);
      });
    }, error => {
      console.error(`Another User is Unable to connect to Room: ${error.message}`);
    });



    // test_time_demoはfirebase consoleで作成ておいてください。
    this.state.db.collection("test_time_demo").get().then((querySnapshot) => {
      querySnapshot.forEach((doc) => {
        //console.log(`${doc.id} => ${doc.data()}`);
        console.log(doc.data())
        console.log(doc.data().changed_at)
        console.log(doc.data().changed_at.toDate())
        this.setState({ date_string: new Date(doc.data().changed_at.toDate()) })
      });
    });
  }

  //
  render() {
    return (
      <div>
        <div> Hello </div>
        <div>日付：
          {this.state.date_string.toString()}
        </div>
      </div>
    )
  }
}


export default App;
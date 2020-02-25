import React from "react";
import cx from "classnames";
import { connect } from "react-redux";
import { setFilter } from "../redux/actions";
import { VISIBILITY_FILTERS } from "../constants";
import { Button,FormGroup, InputGroup,Card, Elevation,ControlGroup} from "@blueprintjs/core";

const VisibilityFilters = ({ activeFilter, setFilter }) => {
  return (
    <div  style={{margin:10}} className="visibility-filters">
      <Card interactive={true} elevation={Elevation.FOUR}>
        {Object.keys(VISIBILITY_FILTERS).map(filterKey => {
          const currentFilter = VISIBILITY_FILTERS[filterKey];
          return (
            <span
              style={{margin:3}}
              key={`visibility-filter-${currentFilter}`}
              className={cx(
                "bp3-button bp3-intent-primary",
                currentFilter === activeFilter && "bp3-intent-danger"
              )}
              onClick={() => {
                setFilter(currentFilter);
              }}
            >
              {currentFilter}
            </span>
          );
        })}
      </Card>
    </div>
  );
};

const mapStateToProps = state => {
  return { activeFilter: state.visibilityFilter };
};
// export default VisibilityFilters;
export default connect(
  mapStateToProps,
  { setFilter }
)(VisibilityFilters);

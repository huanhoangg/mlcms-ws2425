package org.vadere.simulator.models.groups.sir;


import org.vadere.simulator.context.VadereContext;
import org.vadere.annotation.factories.models.ModelClass;
import org.vadere.simulator.models.Model;
import org.vadere.simulator.models.groups.AbstractGroupModel;
import org.vadere.simulator.models.groups.Group;
import org.vadere.simulator.models.groups.GroupSizeDeterminator;
import org.vadere.simulator.models.groups.cgm.CentroidGroup;
import org.vadere.simulator.models.potential.fields.IPotentialFieldTarget;
import org.vadere.simulator.projects.Domain;
import org.vadere.state.attributes.Attributes;
import org.vadere.simulator.models.groups.sir.SIRGroup;
import org.vadere.state.attributes.models.AttributesSIRG;
import org.vadere.state.attributes.scenario.AttributesAgent;
import org.vadere.state.scenario.DynamicElementContainer;
import org.vadere.state.scenario.Pedestrian;
import org.vadere.state.scenario.Topography;
import org.vadere.state.types.PedestrianAttitudeType;
import org.vadere.util.geometry.LinkedCellsGrid;
import org.vadere.util.geometry.shapes.VPoint;
import org.vadere.util.geometry.shapes.VRectangle;

import java.awt.geom.Rectangle2D;
import java.util.*;

/**
 * Implementation of groups for a susceptible / infected / removed (SIR) model.
 */
@ModelClass
public class SIRGroupModel extends AbstractGroupModel<SIRGroup> {

	protected Random random;
	private LinkedHashMap<Integer, SIRGroup> groupsById;
	private Map<Integer, LinkedList<SIRGroup>> sourceNextGroups;
	protected AttributesSIRG attributesSIRG;
	private Topography topography;
	private IPotentialFieldTarget potentialFieldTarget;
	private int totalInfected = 0;
	protected double simTimeStepLength;
	protected static final String simStepLength = "simTimeStepLength";
	private LinkedCellsGrid<Pedestrian> grid;

	public SIRGroupModel() {
		this.groupsById = new LinkedHashMap<>();
		this.sourceNextGroups = new HashMap<>();
	}

	@Override
	public void initialize(List<Attributes> attributesList, Domain domain,
	                       AttributesAgent attributesPedestrian, Random random) {
		this.attributesSIRG = Model.findAttributes(attributesList, AttributesSIRG.class);
		this.topography = domain.getTopography();
		this.random = random;
        this.totalInfected = 0;
		this.simTimeStepLength = VadereContext.getCtx(this.topography).getDouble(simStepLength);
		// define the layout of the grid
		Rectangle2D.Double rect = this.topography.getBounds();
		this.grid = new LinkedCellsGrid<>(new VRectangle(rect), 10); // sideLength = 10 is random choice
	}

	@Override
	public void setPotentialFieldTarget(IPotentialFieldTarget potentialFieldTarget) {
		this.potentialFieldTarget = potentialFieldTarget;
		// update all existing groups
		for (SIRGroup group : groupsById.values()) {
			group.setPotentialFieldTarget(potentialFieldTarget);
		}
	}

	@Override
	public IPotentialFieldTarget getPotentialFieldTarget() {
		return potentialFieldTarget;
	}

	private int getFreeGroupId() {
		if(this.random.nextDouble() < this.attributesSIRG.getInfectionRate()
        || this.totalInfected < this.attributesSIRG.getInfectionsAtStart()) {
			if(!getGroupsById().containsKey(SIRType.ID_INFECTED.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_INFECTED.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_INFECTED.ordinal(), g);
			}
            this.totalInfected += 1;
			return SIRType.ID_INFECTED.ordinal();
		}
		else{
			if(!getGroupsById().containsKey(SIRType.ID_SUSCEPTIBLE.ordinal()))
			{
				SIRGroup g = getNewGroup(SIRType.ID_SUSCEPTIBLE.ordinal(), Integer.MAX_VALUE/2);
				getGroupsById().put(SIRType.ID_SUSCEPTIBLE.ordinal(), g);
			}
			return SIRType.ID_SUSCEPTIBLE.ordinal();
		}
	}


	@Override
	public void registerGroupSizeDeterminator(int sourceId, GroupSizeDeterminator gsD) {
		sourceNextGroups.put(sourceId, new LinkedList<>());
	}

	@Override
	public int nextGroupForSource(int sourceId) {
		return Integer.MAX_VALUE/2;
	}

	@Override
	public SIRGroup getGroup(final Pedestrian pedestrian) {
		SIRGroup group = groupsById.get(pedestrian.getGroupIds().getFirst());
		assert group != null : "No group found for pedestrian";
		return group;
	}

	@Override
	protected void registerMember(final Pedestrian ped, final SIRGroup group) {
		groupsById.putIfAbsent(ped.getGroupIds().getFirst(), group);
	}

	@Override
	public Map<Integer, SIRGroup> getGroupsById() {
		return groupsById;
	}

	@Override
	protected SIRGroup getNewGroup(final int size) {
		return getNewGroup(getFreeGroupId(), size);
	}

	@Override
	protected SIRGroup getNewGroup(final int id, final int size) {
		if(groupsById.containsKey(id))
		{
			return groupsById.get(id);
		}
		else
		{
			return new SIRGroup(id, this);
		}
	}

	private void initializeGroupsOfInitialPedestrians() {
		// get all pedestrians already in topography
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();
		if (c.getElements().size() > 0) {
			// assign the pedestrian to INFECTED if totalInfected is less than InfectionsAtStart, otherwise assign
			// the pedestrian to SUSCEPTIBLE
			for(Pedestrian p : c.getElements()) {
				if (totalInfected < attributesSIRG.getInfectionsAtStart()) {
					assignToGroup(p, SIRType.ID_INFECTED.ordinal());
					totalInfected += 1;
				} else if (totalInfected == attributesSIRG.getInfectionsAtStart()) {
					assignToGroup(p, SIRType.ID_SUSCEPTIBLE.ordinal());
				}
			}
		}
	}

	protected void assignToGroup(Pedestrian ped, int groupId) {
		SIRGroup currentGroup = getNewGroup(groupId, Integer.MAX_VALUE/2);
		currentGroup.addMember(ped);
		ped.getGroupIds().clear();
		ped.getGroupSizes().clear();
		ped.addGroupId(currentGroup.getID(), currentGroup.getSize());
		registerMember(ped, currentGroup);
	}

	protected void assignToGroup(Pedestrian ped) {
		int groupId = getFreeGroupId();
		assignToGroup(ped, groupId);
	}


	/* DynamicElement Listeners */

	@Override
	public void elementAdded(Pedestrian pedestrian) {
		assignToGroup(pedestrian);
	}

	@Override
	public void elementRemoved(Pedestrian pedestrian) {
		Group group = groupsById.get(pedestrian.getGroupIds().getFirst());
		if (group.removeMember(pedestrian)) { // if true pedestrian was last member.
			groupsById.remove(group.getID());
		}
	}

	/* Model Interface */

	@Override
	public void preLoop(final double simTimeInSec) {
		initializeGroupsOfInitialPedestrians();
		topography.addElementAddedListener(Pedestrian.class, this);
		topography.addElementRemovedListener(Pedestrian.class, this);
	}

	@Override
	public void postLoop(final double simTimeInSec) {
	}

	private int last_simTime_int = -1;  // a variable for the update function below

	@Override
	public void update(final double simTimeInSec) {
		// compare the current integer part of simTimeInSec to the last recorded integer part of simTimeInSec, if the
		// former is bigger than the latter, then one second has passed and we execute the update procedure; otherwise
		// if both are equal, then we are still in the same second, and the update function is to be skipped
		int current_simTime_int = (int) simTimeInSec;
		if (current_simTime_int == last_simTime_int) {
			return;
		}
		DynamicElementContainer<Pedestrian> c = topography.getPedestrianDynamicElements();
		if (c.getElements().size() > 0) {
			// register all the pedestrians into the grid
			for (Pedestrian p : c.getElements()) {
				this.grid.addObject(p);
			}
			// execute the Group Update logic based on the current group of the pedestrian
			for(Pedestrian p : c.getElements()) {
				SIRGroup g = getGroup(p);
				// process the transition infected -> recovery
				if (g.getID() == SIRType.ID_INFECTED.ordinal()) {
					if (this.random.nextDouble() < attributesSIRG.getRecoveryRate()) {
						elementRemoved(p);
						assignToGroup(p, SIRType.ID_REMOVED.ordinal());
						this.totalInfected -= 1;
					}
					continue;
				}
				// skip this pedestrian if it is already in the REMOVED group
				if (g.getID() == SIRType.ID_REMOVED.ordinal()) {
					continue;
				}
				// get pedestrian position and the list of neighbors within the radius of InfectionMaxDistance
				VPoint ped_pos = p.getPosition();
				List<Pedestrian> peds_nearby = grid.getObjects(ped_pos, attributesSIRG.getInfectionMaxDistance());
				// loop over the neighbors and set infected according to the InfectionRate
				for(Pedestrian p_neighbor : peds_nearby) {
					if(p == p_neighbor || getGroup(p_neighbor).getID() != SIRType.ID_INFECTED.ordinal())
						continue;
					if (this.random.nextDouble() < attributesSIRG.getInfectionRate()) {
						elementRemoved(p);
						assignToGroup(p, SIRType.ID_INFECTED.ordinal());
						break;
					}
				}
			}
		}
		// clear the grid for registering the pedestrians into the grid at next update step
		for (Pedestrian p : c.getElements()) {
			this.grid.removeObject(p);
		}
		last_simTime_int = current_simTime_int;
	}
}